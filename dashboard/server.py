"""Read-only, loopback-only training dashboard. No torch or build dependencies."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import mimetypes
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / 'static'
FIELDS = ('step', 'loss', 'learning_rate', 'grad_norm', 'cuda_peak_allocated_bytes',
          'cuda_peak_reserved_bytes', 'elapsed_seconds', 'hidden_nmse', 'key_nmse',
          'value_nmse', 'next_token_kl', 'expert_velocity_nmse', 'expert_probe_index')


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return default


def read_trace(path):
    rows, warnings = [], []
    try:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
    except OSError:
        return rows, warnings
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            # A writer may currently be completing the last JSONL record.
            warnings.append('末行尚未写完' if index == len(lines)-1 else f'第{index+1}行记录无法解析')
    return rows, warnings


def safe_number(value):
    return value if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) else None


def clip_means(records):
    grouped = {}
    for row in records:
        if safe_number(row.get('ADE_m')) is not None:
            grouped.setdefault(row['clip_id'], []).append(row)
    return [{'clip_id': clip, 'ADE_m': sum(r['ADE_m'] for r in group)/len(group),
             'teacher_ADE_m': sum(r['dense_ADE_m'] for r in group)/len(group)
             if all(safe_number(r.get('dense_ADE_m')) is not None for r in group) else None,
             'seeds': len(group)} for clip, group in sorted(grouped.items())]


def load_experiment(path):
    provenance = read_json(path/'provenance.json', {})
    rows, warnings = read_trace(path/'training_trace.jsonl')
    training = [{k: safe_number(r.get(k)) for k in FIELDS} for r in rows if r.get('phase') == 'training']
    monitoring = []
    for r in rows:
        if r.get('phase') != 'validation':
            continue
        metrics = r.get('metrics', [])
        values = {k: sum(m[k] for m in metrics)/len(metrics) for k in FIELDS[7:12]
                  if metrics and all(safe_number(m.get(k)) is not None for m in metrics)}
        monitoring.append({'step': r['step'], 'loss': safe_number(r.get('loss')), **values})
    tasks = []
    for directory in sorted((path/'task_validation').glob('step_*')):
        result = read_json(directory/'result.json')
        progress = result or read_json(directory/'progress.json', {})
        try:
            step = int(directory.name.split('_')[-1])
        except ValueError:
            continue
        records = progress.get('records', [])
        failures = progress.get('failures', [])
        # Never turn a partially completed validation into a full ADE.
        complete = bool(result) and len(records) == 60 and not failures
        value = safe_number(result.get('mean_ADE_m', result.get('mean_minADE_m'))) if complete else None
        tasks.append({'step': step, 'ADE_m': value, 'complete': complete,
                      'records': len(records), 'failures': len(failures),
                      'status': 'completed' if complete else 'invalid' if result else 'evaluating',
                      'clips': clip_means(records) if complete else [],
                      'mlp_tensor_sha256': result.get('mlp_tensor_sha256') if result else None})
    recovery = read_json(path/'controlled_recovery_result.json', {})
    post = read_json(Path(str(path)+'.postprocess.json'), {})
    interruption = read_json(path/'interruption_reason.json', {})
    result = read_json(path/'result.json', {})
    candidate = (result.get('candidates') or [{}])[0]
    valid = [t for t in tasks if t['ADE_m'] is not None]
    best = min(valid, key=lambda t: t['ADE_m']) if valid else None
    status = 'completed' if post.get('status') == 'completed' or recovery.get('status') == 'completed' else 'running'
    if post.get('status') == 'failed' or interruption:
        status = 'interrupted' if interruption else 'failed'
    checkpoint_manifest = read_trace(path/'adapter_checkpoints/manifest.jsonl')[0]
    newest = [path/'training_trace.jsonl', path/'provenance.json']
    newest += list((path/'task_validation').glob('*/progress.json'))
    updated = max((f.stat().st_mtime for f in newest if f.exists()), default=path.stat().st_mtime)
    split = read_json(path/'optimization_split.json', {})
    started = read_json(path/'training_started_verified.json', {})
    params = recovery.get('params') or candidate.get('params') or started.get('params')
    if not params:
        parent = read_json(Path(provenance.get('parent_experiment', '__missing__'))/'controlled_recovery_result.json', {})
        params = parent.get('params') if not provenance.get('new_pruning_allocation_sha256') else None
    selected_step = recovery.get('selected_step', best['step'] if best else None)
    selected = next((t for t in tasks if t['step'] == selected_step and t['complete']), best)
    teacher = recovery.get('dense_mean_ADE_m')
    if teacher is None:
        ref = read_json(path/'task_validation_reference.json', {})
        teacher = ref.get('dense_mean_ADE_m')
    return {'id': path.name, 'path': str(path), 'status': status, 'updated_at': updated,
            'params': params, 'steps': provenance.get('steps'), 'current_step': training[-1]['step'] if training else 0,
            'eval_every': provenance.get('eval_every'), 'save_every': provenance.get('save_adapters_every'),
            'learning_rate': provenance.get('lr'), 'rank': provenance.get('rank'), 'seed': provenance.get('seed'),
            'gpu': provenance.get('gpu'), 'started_at': provenance.get('started_at_utc'),
            'training': training, 'monitoring': monitoring, 'task_validation': tasks,
            'best_ADE_m': best['ADE_m'] if best else None, 'best_step': best['step'] if best else None,
            'selected_step': selected_step, 'clip_comparison': selected['clips'] if selected else [],
            'teacher_ADE_m': teacher, 'precision_pass': recovery.get('conditional_15pct_pass'),
            'checkpoint_count': len(checkpoint_manifest), 'checkpoint_bytes': sum(r.get('bytes', 0) for r in checkpoint_manifest),
            'training_clips': len(split.get('training', [])) or None,
            'monitor_clips': len(split.get('monitor_validation', [])) or None,
            'checkpoint_revision': provenance.get('checkpoint_revision'),
            'dataset_revision': provenance.get('dataset_revision'),
            'validation_identity': read_json(path/'task_validation_reference.json', {}).get('identity'),
            'warnings': warnings, 'interruption': interruption.get('reason'),
            'metric_semantics': '固定20段；每段3个配对噪声种子的平均ADE，不是best-of-three minADE',
            'ADE_is_every_training_step': False}


class Repository:
    def __init__(self, root, remote=None):
        self.root, self.remote = root, remote
        self.cache_path = root/'dashboard/cache/last_server_snapshot.json'
        self.remote_cache = read_json(self.cache_path) if remote else None
        self.snapshot = None
        self.until = 0
        self.lock = threading.Lock()

    def local_experiments(self):
        folders = list((self.root/'experiments').glob('*'))
        folders += list((self.root/'compression/results').glob('*/experiments/*'))
        unique = {}
        for path in folders:
            if path.is_dir() and (path/'training_trace.jsonl').exists():
                reference = read_json(path/'task_validation_reference.json', {})
                if len(reference.get('clips', [])) != 20:
                    continue  # Keep this dashboard's validation protocol explicit.
                if path.name not in unique or path.stat().st_mtime > unique[path.name].stat().st_mtime:
                    unique[path.name] = path
        return [dict(load_experiment(p), source='local_archive') for p in unique.values()]

    def get(self):
        with self.lock:
            if self.snapshot and time.monotonic() < self.until:
                return self.snapshot
            experiments = {e['id']: e for e in self.local_experiments()}
            connection, error = ('local', None)
            if self.remote:
                try:
                    request = urllib.request.Request(self.remote.rstrip('/')+'/api/snapshot', headers={'Accept-Encoding':'gzip'})
                    with urllib.request.urlopen(request, timeout=10) as response:
                        raw = response.read()
                        if response.headers.get('Content-Encoding') == 'gzip':
                            raw = gzip.decompress(raw)
                        remote = json.loads(raw)
                    self.remote_cache = remote
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.cache_path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(remote, ensure_ascii=False), encoding='utf-8')
                    temporary.replace(self.cache_path)
                    connection = 'live'
                except (OSError, ValueError, urllib.error.URLError) as exc:
                    connection, error = 'offline', str(exc)
                if self.remote_cache:
                    for e in self.remote_cache['experiments']:
                        experiments[e['id']] = dict(e, source='server' if connection == 'live' else 'server_cached')
            else:
                # Server-side instance reads native experiment files directly.
                for e in experiments.values():
                    if (self.root/'experiments'/e['id']).exists():
                        e['source'] = 'server'
            items = sorted(experiments.values(), key=lambda e: e.get('started_at') or e['id'], reverse=True)
            items.sort(key=lambda e: e['status'] != 'running')
            self.snapshot = {'experiments': items, 'connection': connection, 'connection_error': error,
                             'fetched_at': time.time(), 'remote_fetched_at': self.remote_cache.get('fetched_at') if self.remote_cache else None,
                             'refresh_interval_seconds': 5, 'timezone': 'Asia/Shanghai'}
            self.until = time.monotonic()+2
            return self.snapshot


def make_handler(repository):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            if url.path == '/api/snapshot':
                try:
                    payload = repository.get()
                    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
                except Exception:
                    self.send_error(500, 'Cannot read experiment records')
                    return
                self.send_response(200)
                compressed = 'gzip' in self.headers.get('Accept-Encoding', '')
                if compressed:
                    data = gzip.compress(data, compresslevel=1)
                    self.send_header('Content-Encoding', 'gzip')
                    self.send_header('Vary', 'Accept-Encoding')
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # Browser closed a request; leave training and service running.
                return
            relative = urllib.parse.unquote(url.path).lstrip('/') or 'index.html'
            target = (STATIC/relative).resolve()
            if not target.is_relative_to(STATIC.resolve()) or not target.is_file():
                self.send_error(404)
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--remote', help='Loopback SSH-forwarded API URL')
    args = parser.parse_args()
    if args.remote and urllib.parse.urlsplit(args.remote).hostname not in ('127.0.0.1', 'localhost'):
        parser.error('Remote API must be an existing loopback SSH forward')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(Repository(args.root.resolve(), args.remote)))
    print(f'Dashboard ready at http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
