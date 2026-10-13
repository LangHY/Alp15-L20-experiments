"""一次性七组 teacher-cache 来源范围查询。仅文本及 PT lstat。"""
import csv
import hashlib
import io
import json
import os
import platform
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/tuqiang/lang/Lighten_VLA')
CENSUS = ROOT / 'experiments/precision_holdout_prep_20261010_resume_v3'
GROUPS = {
    'mlp_measured6p8_recovery_20261002_190306': 'bc5e3999b6416895b83c91aaef46178d3a596a2316daeffd24e4c3be86c045cd',
    'mlp_measured7_recovery_20261002_181113': '69801df440a77311b355ff8cf4a42365ed06c9aab04eecff186e33a20c14a785',
    'mlp_measured7_recovery_20261002_182132': 'bc5e3999b6416895b83c91aaef46178d3a596a2316daeffd24e4c3be86c045cd',
    'mlp_newdata8_recovery_20261002_165238': 'f5f95930b99c3b31547d16acc01be992c2e6fd67d81fd6d81d5c56023585d457',
    'mlp_precision8_balanced_20261002_195311': '60b6f7d78cdb762cb5cb0852e2783974d7e7485d248fe45d47fe3924abe0639f',
    'mlp_precision8_dense_20261002_195955': '232589eb5a3d6357e865af278e83ffbbca3fbc786932d12a17a8614b2b579771',
    'mlp_rebalanced7_recovery_20261002_210458': '1106c9488b8b05e11cd7212b4f722631ea803c0c80d05f3ba17b9f7d30da3850',
}
AUDIT_SHA = '472a5d107e3876ce0e81727519c0c67b7c1e6f44fdb6aae90814ce32ecbda349'
before = {}


def read(path):
    assert path.is_relative_to(ROOT / 'experiments')
    assert path.suffix in {'.json', '.csv', '.py'}
    assert stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    raw = path.read_bytes()
    before[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8', errors='strict')


def flag_path(argv, name):
    assert argv.count(name) == 1
    p = Path(argv[argv.index(name) + 1])
    p = p if p.is_absolute() else ROOT / p
    assert '..' not in p.parts and p.is_relative_to(ROOT / 'experiments')
    return p


started = datetime.now(timezone.utc).isoformat()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1'
ledger = json.loads(read(CENSUS / 'usage-ledger.json'))
assert before[str(CENSUS / 'usage-ledger.json')] == '0697cec28cd36ca841fe75ddeb95f1037269d69871bc2bda59ae831947e870d1'
rows = list(csv.DictReader(io.StringIO(read(CENSUS / 'unresolved-assets.csv'))))
assert before[str(CENSUS / 'unresolved-assets.csv')] == 'f398e3f054620d37544217e766fbafba2d6d052f084df75e22df9f41259a09c7'
groups, files = [], set()
for name, sha in GROUPS.items():
    out = ROOT / 'experiments' / name
    provenance = json.loads(read(out / 'provenance.json'))
    source_path = out / 'code/run_interface_recovery.py'
    source = read(source_path)
    assert before[str(source_path)] == sha == provenance['code_sha256'][source_path.name]
    audit = json.loads(read(out / 'new_data_audit.json'))
    split = json.loads(read(out / 'optimization_split.json'))
    argv = provenance['argv']
    audit_path = flag_path(argv, '--new-training-audit') / 'result.json'
    original_audit = json.loads(read(audit_path))
    assert before[str(audit_path)] == before[str(out / 'new_data_audit.json')] == AUDIT_SHA
    assert provenance['new_data_audit_sha256'] == split['data_audit_sha256'] == AUDIT_SHA
    assert original_audit == audit and audit['status'] == 'completed'
    assert audit['dataset_revision'] == provenance['dataset_revision'] == ledger['dataset_revision']
    assert '--training-window-audit' not in argv and '--resume' not in argv
    role_records = {'calibration': audit['training_records'], 'validation': audit['monitor_records']}
    role_ids = {role: [row['clip_id'] for row in records] for role, records in role_records.items()}
    assert len(role_ids['calibration']) == len(set(role_ids['calibration'])) == 80
    assert len(role_ids['validation']) == len(set(role_ids['validation'])) == 20
    assert not set(role_ids['calibration']) & set(role_ids['validation'])
    assert set(split['training']) == set(role_ids['calibration'])
    assert set(split['monitor_validation']) == set(role_ids['validation'])
    assert split['gradient_updates_on_monitor'] is False
    all_ids = set(role_ids['calibration']) | set(role_ids['validation'])
    assert {row['clip_id'] for row in audit['records']} == all_ids
    assert all_ids <= set(ledger['used_clip_ids'])
    donor_evidence = None
    if '--teacher-interface-cache' in argv:
        donor = flag_path(argv, '--teacher-interface-cache')
        assert donor.name in GROUPS
        reused = provenance['reused_teacher_cache']
        assert reused['directory'] == str(donor)
        donor_provenance = json.loads(read(donor / 'provenance.json'))
        donor_split = json.loads(read(donor / 'optimization_split.json'))
        donor_audit = json.loads(read(donor / 'new_data_audit.json'))
        assert before[str(donor / 'new_data_audit.json')] == AUDIT_SHA
        assert donor_provenance['new_data_audit_sha256'] == AUDIT_SHA
        for key in ('source_commit', 'checkpoint_revision', 'dataset_revision', 'logit_weight', 'expert_weight'):
            assert donor_provenance[key] == provenance[key]
        assert donor_provenance['rollout_prefix'] is True
        assert donor_audit == audit
        assert set(donor_split['training']) == set(role_ids['calibration'])
        assert set(donor_split['monitor_validation']) == set(role_ids['validation'])
        expected_donor = {str(donor / 'teacher_cache' / f'{role}_{index}.pt')
                          for role, records in role_records.items() for index in range(len(records))}
        assert set(reused['source_sha256']) == expected_donor
        assert all(len(value) == 64 and all(c in '0123456789abcdef' for c in value)
                   for value in reused['source_sha256'].values())
        donor_evidence = {'experiment': donor.name, 'declared_source_files': len(expected_donor),
                          'declared_payload_sha_verified': False}
    else:
        assert name == 'mlp_newdata8_recovery_20261002_165238'
        assert not provenance.get('reused_teacher_cache')
    expected = {f'{role}_{index}.pt' for role, records in role_records.items()
                for index in range(len(records))}
    folder = out / 'teacher_cache'
    assert folder.is_dir() and not folder.is_symlink()
    assert {p.name for p in folder.iterdir() if p.suffix == '.pt'} == expected
    file_stats = []
    for role, records in role_records.items():
        for index in range(len(records)):
            path = folder / f'{role}_{index}.pt'
            info = path.lstat()
            assert stat.S_ISREG(info.st_mode) and not path.is_symlink()
            relative = str(path.relative_to(ROOT))
            assert relative not in files
            files.add(relative)
            file_stats.append({'source': relative, 'role': role, 'index': index,
                               'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns})
    groups.append({'experiment': name, 'producer_sha256': sha, 'producer_source': source,
                   'ordered_role_clip_ids': role_ids, 'file_stats': file_stats,
                   'donor_evidence': donor_evidence, 'status': 'candidate_bounded_used_group',
                   'scope': '100-clip group lineage; no payload identity verification',
                   'payload_verified': False})
assert len(files) == 700
selected_rows = [row for row in rows if row['source'] in files and row['reason'] in {
    'anonymous_cache_without_explicit_local_mapping', 'binary_payload_not_verified'}]
counts = Counter(row['reason'] for row in selected_rows)
assert counts == Counter({'anonymous_cache_without_explicit_local_mapping': 698,
                         'binary_payload_not_verified': 700})
for reason in counts:
    selected = [row['source'] for row in selected_rows if row['reason'] == reason]
    assert len(selected) == len(set(selected))
for group in groups:
    for record in group['file_stats']:
        info = (ROOT / record['source']).lstat()
        assert stat.S_ISREG(info.st_mode)
        assert info.st_size == record['bytes'] and info.st_mtime_ns == record['mtime_ns']
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in before.items())
print(json.dumps({'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'python': platform.python_version(), 'query_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'input_text_sha256': before, 'groups': groups, 'selected_original_csv_rows': selected_rows,
    'original_record_counts': dict(Counter(row['reason'] for row in rows)),
    'candidate_record_counts': dict(counts), 'unique_teacher_files': 700,
    'unique_already_used_clip_ids': sorted(set(groups[0]['ordered_role_clip_ids']['calibration']) |
                                        set(groups[0]['ordered_role_clip_ids']['validation'])),
    'input_texts_unchanged': True, 'payload_verified': False, 'new_excluded_chunks': [],
    'coverage_complete': False, 'drive_disjoint_verified': False, 'exit_code': 0},
    ensure_ascii=False, indent=2))
