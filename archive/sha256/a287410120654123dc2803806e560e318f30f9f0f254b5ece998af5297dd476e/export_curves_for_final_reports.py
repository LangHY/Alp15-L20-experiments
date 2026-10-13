"""Export recorded local distillation and paired trajectory evaluation curves.

No metric is inferred from a configured learning rate or a missing step. Local
MLP NMSE is never labeled end-to-end loss. This reads immutable JSON snapshots;
rerun it as running experiments produce additional records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
import numpy as np

LANGUAGE = 'en'
CJK_FONT = None


def _chinese(text):
    replacements = [
        ('mlp_interface7_expert_20261002_001500', '7.228B 冻结 Expert 动作蒸馏'),
        ('mlp_interface7_logits_20261001_235500', '7.228B 接口与 token 联合蒸馏'),
        ('mlp_interface7_20261001_234500', '7.228B 整网 hidden/KV 蒸馏'),
        ('mlp_prune_distill_20261001_171143', '10B 单投影蒸馏'),
        ('mlp_stage8_joint_20261001_182631', '8B 联合蒸馏'),
        ('mlp_stage8_20261001_175032', '8B 单投影蒸馏'),
        ('mlp_stage6_joint_20261001_184023', '6B 联合蒸馏'),
        ('mlp_layerwise8_20261001_220500', '非均匀 8B 联合蒸馏'),
        ('mlp_layerwise7_20261001_220500', '非均匀 7.228B 联合蒸馏'),
        ('Recorded samples only; no interpolation or end-to-end loss', '实际采样点；指标为局部特征损失'),
        ('local cached-teacher MLP output NMSE', '基于教师缓存的局部 MLP 误差（NMSE）'),
        ('baseline, best observed, and final export', '蒸馏前、最佳验证与最终导出对照'),
        ('Local exported-dtype validation NMSE', '实际导出精度的验证误差（NMSE）'),
        ('Local MLP NMSE', '局部 MLP 误差（NMSE）'),
        ('Recorded optimizer step', '优化步数（实际采样）'),
        ('Untrained baseline', '剪枝后、蒸馏前'),
        ('Best observed validation', '最佳验证结果'),
        ('Final exported weights', '最终导出权重'),
        ('Export validation', '导出权重验证损失'), ('Selected export', '选中权重'),
        ('Train batch', '训练批次损失'), ('Not recorded', '未采集'),
        ('Student ADE - dense ADE (m)', '学生减教师的 ADE 差（米）'),
        ('Mean minADE (m)', '平均轨迹误差 ADE（米）'),
        ('Total parameters (billions)', '总参数量（十亿）'),
        ('paired trajectory evaluation', '配对轨迹评估'),
        ('paired ADE differences', '配对轨迹误差差值'),
        ('internal single-chunk clips', '单一训练分块内的片段'),
        ('recorded model comparisons', '已记录的模型对照'),
        ('expanded_validation', '扩展验证集'), ('expanded_holdout', '扩展留出集'),
        ('expanded validation', '扩展验证集'), ('expanded holdout', '扩展留出集'),
        ('validation', '验证集'), ('Dense teacher', '原始教师'), ('Student', '学生模型'),
        ('stage8 joint', '8B 联合蒸馏'), ('stage8', '8B 单投影蒸馏'),
        ('stage6 joint', '6B 联合蒸馏'), ('student', '学生模型'),
        ('layerwise8', '非均匀 8B 联合蒸馏'), ('layerwise7', '非均匀 7.228B 联合蒸馏'),
        ('recorded', '实测'), ('width_layerwise_5376', '非均匀宽度'),
        ('width_layerwise_3584', '非均匀宽度'),
        ('minADE (m)', 'ADE（米）'), ('ADE ECDF', 'ADE 经验累积分布'),
        ('ECDF', '经验累积分布'), ('Clips', '片段数'), ('MLP layer', 'MLP 层编号'),
        ('Status:', '状态：'), ('completed', '完成'), ('n=', '样本数='),
        ('lr', '学习率'), ('gradient_norm', '梯度范数（裁剪前）'),
        ('elapsed_seconds', '累计耗时（秒）'),
        ('cuda_allocated_bytes', '已分配显存（字节）'),
        ('cuda_reserved_bytes', '已预留显存（字节）'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return re.sub(r'\bLayer (\d+)\b', r'第 \1 层', text)

METRICS = ('val_nmse', 'train_batch_nmse', 'lr', 'gradient_norm', 'elapsed_seconds',
           'cuda_allocated_bytes', 'cuda_reserved_bytes')
OPTIONAL = ('lr', 'gradient_norm', 'elapsed_seconds', 'cuda_allocated_bytes', 'cuda_reserved_bytes')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _safe(value):
    return re.sub(r'[^a-zA-Z0-9_-]+', '_', value)


class Snapshots:
    def __init__(self):
        self.cache, self.inputs, self.warnings = {}, [], []

    def read(self, path):
        path = path.resolve()
        if path in self.cache:
            return self.cache[path]
        try:
            data = path.read_bytes()
            payload = json.loads(data)
        except (OSError, ValueError) as exc:
            self.warnings.append({'file': str(path), 'error': str(exc),
                                  'note': 'Skipped unreadable/incomplete snapshot; retry export later'})
            self.cache[path] = None
            return None
        self.cache[path] = payload
        self.inputs.append({'file': str(path), 'sha256': _sha(data), 'bytes': len(data)})
        return payload

    def lines(self, path):
        path = path.resolve()
        if path in self.cache:
            return self.cache[path]
        try:
            data = path.read_bytes()
        except OSError as exc:
            self.warnings.append({'file': str(path), 'error': str(exc)})
            return []
        events = []
        for number, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError('JSONL event is not an object')
                events.append({'line': number, 'event': event})
            except ValueError as exc:
                self.warnings.append({'file': str(path), 'line': number, 'error': str(exc),
                                      'note': 'Incomplete/invalid event skipped; no invented replacement'})
        self.inputs.append({'file': str(path), 'sha256': _sha(data), 'bytes': len(data)})
        self.cache[path] = events
        return events


def _csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, output, name, manifest, kind, **metadata):
    if LANGUAGE == 'zh':
        for item in fig.findobj(match=Text):
            item.set_text(_chinese(item.get_text()))
            item.set_fontproperties(font_manager.FontProperties(fname=CJK_FONT, size=item.get_fontsize()))
    files = []
    for extension in ('png', 'svg'):
        path = output / f'{name}.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        files.append({'file': path.name, 'sha256': _sha(path.read_bytes())})
    plt.close(fig)
    manifest['charts'].append({'kind': kind, 'files': files, **metadata})


def _interface_plots(root, prefix, snapshots, output, manifest):
    path = root / 'training_trace.jsonl'
    if not path.is_file():
        return False
    rows = [item['event'] for item in snapshots.lines(path)]
    for row in rows:
        if row.get('phase') == 'validation' and row.get('metrics'):
            for key in ('hidden_nmse', 'key_nmse', 'value_nmse', 'next_token_kl', 'expert_velocity_nmse'):
                if all(key in m for m in row['metrics']):
                    row[key] = sum(m[key] for m in row['metrics']) / len(row['metrics'])
    train = [r for r in rows if r.get('phase') == 'training']
    validation = [r for r in rows if r.get('phase') == 'validation']
    columns = ['phase', 'step', 'clip_id', 'loss', 'hidden_nmse', 'key_nmse', 'value_nmse', 'next_token_kl', 'expert_velocity_nmse',
               'learning_rate', 'grad_norm', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes', 'elapsed_seconds']
    _csv(output / f'{prefix}_interface_trace.csv', rows, columns)
    has_kl = any('next_token_kl' in r for r in train)
    has_expert = any('expert_velocity_nmse' in r for r in train)
    panels = 2 + int(has_kl) + int(has_expert)
    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 4.5), constrained_layout=True)
    for items, label, color in ((train, '训练片段', '#276787'), (validation, '固定验证集', '#c66928')):
        if items:
            axes[0].scatter([r['step'] for r in items], [r['loss'] for r in items], label=label, color=color, s=16)
    for metric, label in (('hidden_nmse', '最终 hidden'), ('key_nmse', '所有层 Key'), ('value_nmse', '所有层 Value')):
        if train:
            axes[1].scatter([r['step'] for r in train], [r[metric] for r in train], label=label, s=12)
    if has_kl:
        for items, label in ((train, '训练片段'), (validation, '固定验证集')):
            observed = [r for r in items if 'next_token_kl' in r]
            if observed:
                axes[2].scatter([r['step'] for r in observed], [r['next_token_kl'] for r in observed], label=label, s=12)
    if has_expert:
        for items, label in ((train, '训练片段单探针'), (validation, '固定验证集四探针')):
            observed = [r for r in items if 'expert_velocity_nmse' in r]
            if observed:
                axes[2 + int(has_kl)].scatter([r['step'] for r in observed],
                    [r['expert_velocity_nmse'] for r in observed], label=label, s=12)
    for ax in axes:
        ax.set_xlabel('实际优化步数'); ax.set_ylabel('整网接口相对均方误差'); ax.legend(); ax.grid(alpha=0.2)
    axes[0].set_ylabel('蒸馏总损失（各项损失按配置加权）')
    if has_kl:
        axes[2].set_ylabel('下一 token 分布 KL 散度')
    if has_expert:
        axes[2 + int(has_kl)].set_ylabel('冻结 Expert 的速度场相对均方误差')
    fig.suptitle('完整学生图蒸馏：接口与配置的辅助损失（教师强制前缀）')
    _save(fig, output, f'{prefix}_interface_loss', manifest, 'whole_student_interface_loss',
          semantics='Hidden NMSE + mean(Key NMSE, Value NMSE) + configured logit_weight * next-token KL + configured expert_weight * velocity NMSE; not trajectory loss')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metrics, title in (
        (axes[0, 0], [('learning_rate', '学习率', 1)], '学习率'),
        (axes[0, 1], [('grad_norm', '裁剪前梯度范数', 1)], '梯度范数'),
        (axes[1, 0], [('cuda_peak_allocated_bytes', '峰值已分配', 2**30),
                      ('cuda_peak_reserved_bytes', '峰值已保留', 2**30)], 'PyTorch 显存（GiB）'),
        (axes[1, 1], [('elapsed_seconds', '整网蒸馏累计耗时', 1)], '整网蒸馏累计耗时（秒）')):
        for key, label, divisor in metrics:
            pairs = [(r['step'], r[key] / divisor) for r in train if _finite(r.get(key))]
            if pairs:
                ax.scatter(*zip(*pairs), label=label, s=12)
        ax.set_title(title); ax.set_xlabel('实际优化步数'); ax.grid(alpha=0.2)
        if train:
            ax.legend()
    _save(fig, output, f'{prefix}_interface_resources', manifest, 'whole_student_interface_resources')
    manifest['sampling'].append({'file': str(path), 'training_steps': [r['step'] for r in train],
        'validation_steps': [r['step'] for r in validation], 'note': 'Actual samples only; training clip alternates, no smoothing'})
    manifest['metric_semantics']['whole_student_interface_loss'] = 'hidden NMSE + (Key NMSE + Value NMSE)/2; not trajectory ADE'
    manifest['metric_semantics']['distillation_total_loss'] = 'Interface loss plus auxiliary KL and Expert velocity losses weighted by experiment provenance'
    manifest['metric_semantics']['interface_elapsed_seconds'] = 'Whole-student optimizer loop plus its validation; excludes teacher collection, local prefit and cold initialization'
    manifest['metric_semantics']['expert_velocity_nmse'] = 'Frozen Expert velocity probes; one rotating training probe, mean of four fixed validation probes; not trajectory ADE'
    audit_path = root / 'merged_interface_audit.json'
    if audit_path.is_file():
        audit = snapshots.read(audit_path)
        if audit and audit.get('status') == 'completed':
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
            for ax, role, title in zip(axes, ('calibration', 'validation'), ('校准片段', '固定验证片段')):
                for j, (key, label) in enumerate((('parent', '恢复前'), ('merged_bf16', '合并 BF16 后'))):
                    records = audit['models'][key][role]['records']
                    values = [sum(r[m] for r in records) / len(records) for m in ('hidden_nmse', 'key_nmse', 'value_nmse')]
                    ax.bar(np.arange(3) + (j - 0.5) * 0.35, values, width=0.35, label=label)
                ax.set_xticks(np.arange(3), ['最终 hidden', '所有层 Key', '所有层 Value'])
                ax.set_title(title); ax.set_ylabel('相对均方误差'); ax.legend(); ax.grid(axis='y', alpha=0.2)
            fig.suptitle('新进程验证：实际 BF16 合并权重的接口误差')
            _save(fig, output, f'{prefix}_merged_interface_audit', manifest, 'merged_bf16_interface_comparison',
                  selection_uses_this_report=False)
    return True


def _nearest_split(directory, root, snapshots):
    current = directory
    while current == root or root in current.parents:
        path = current / 'split.json'
        if path.is_file():
            split = snapshots.read(path)
            if isinstance(split, dict):
                canonical = json.dumps(split, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
                return split, _sha(canonical.encode('utf-8')), str(path)
        if current == root:
            break
        current = current.parent
    return None, None, None


def _observed(rows, metric):
    return [(row['step'], row[metric]) for row in rows
            if _finite(row.get('step')) and _finite(row.get(metric))]


def _intervals(rows, metric):
    steps = sorted(set(step for step, _ in _observed(rows, metric)))
    intervals = Counter(b - a for a, b in zip(steps, steps[1:]))
    return {'recorded_steps': steps, 'observed_step_intervals':
            [{'interval': interval, 'count': count} for interval, count in sorted(intervals.items())]}


def _layer_plots(group, output, manifest):
    reports, rows = group['reports'], group['rows']
    by_layer = {layer: [row for row in rows if row['layer'] == layer] for layer in range(36)}
    fig, axes = plt.subplots(6, 6, figsize=(20, 16), constrained_layout=True)
    for layer, ax in enumerate(axes.flat):
        items = by_layer[layer]
        step_rows = [row for row in items if row['record_type'] == 'step_history']
        if not step_rows:
            step_rows = [row for row in items if row['record_type'] == 'jsonl_step']
        history_rows = [row for row in items if row['record_type'] == 'history']
        if not history_rows:
            history_rows = [row for row in items if row['record_type'] in ('jsonl_baseline', 'jsonl_step')]
        training = _observed(step_rows, 'train_batch_nmse') or _observed(history_rows, 'train_batch_nmse')
        validation = _observed(history_rows, 'val_nmse')
        if not validation:
            validation = _observed(step_rows, 'val_nmse')
        for pairs, label, color in ((training, 'Train batch', '#276787'),
                                    (validation, 'Export validation', '#c66928')):
            if pairs:
                x, y = zip(*pairs)
                ax.scatter(x, y, s=8, alpha=0.85, label=label, color=color)
        if not training and not validation:
            ax.text(0.5, 0.5, 'Not recorded', ha='center', va='center', transform=ax.transAxes, color='#777777')
        report = reports.get(layer, {})
        if _finite(report.get('best_step')) and _finite(report.get('final_val_nmse')):
            ax.scatter([report['best_step']], [report['final_val_nmse']], marker='*',
                       color='#3e8555', s=45, label='Selected export')
        ax.set_title(f'Layer {layer}', fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        if layer >= 30:
            ax.set_xlabel('Recorded optimizer step', fontsize=8)
        if layer % 6 == 0:
            ax.set_ylabel('Local MLP NMSE', fontsize=8)
    handles, labels = [], []
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc='upper right', ncol=3, fontsize=9)
    fig.suptitle(f'{group["label"]}: local cached-teacher MLP output NMSE\n'
                 'Recorded samples only; no interpolation or end-to-end loss', fontsize=15)
    _save(fig, output, group['name'] + '_layer_loss', manifest, '36_layer_local_loss',
          experiment=group['experiment'], candidate=group['candidate'], layers_recorded=sorted(reports))
    summary_rows = []
    for layer, report in sorted(reports.items()):
        history_values = [value for _, value in _observed(by_layer.get(layer, []), 'val_nmse')]
        explicit_best = report.get('best_val_nmse')
        # This is explicitly the minimum observed validation point, not a newly
        # invented training event. Final export NMSE remains its separate field.
        best = explicit_best if _finite(explicit_best) else min(history_values) if history_values else None
        summary_rows.append({'experiment': group['experiment'], 'candidate': group['candidate'],
                             'layer': layer, 'baseline_val_nmse': report.get('baseline_val_nmse'),
                             'best_observed_val_nmse': best, 'final_val_nmse': report.get('final_val_nmse'),
                             'best_step': report.get('best_step'),
                             'baseline_train_nmse': report.get('baseline_train_nmse'),
                             'final_train_nmse': report.get('final_train_nmse')})
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for field, label, color, marker in (
        ('baseline_val_nmse', 'Untrained baseline', '#777777', 'o'),
        ('best_observed_val_nmse', 'Best observed validation', '#3e8555', '*'),
        ('final_val_nmse', 'Final exported weights', '#c66928', 'x')):
        points = [(row['layer'], row[field]) for row in summary_rows if _finite(row[field])]
        if points:
            x, y = zip(*points)
            ax.scatter(x, y, label=label, color=color, marker=marker, s=40)
    ax.set(xlabel='MLP layer', ylabel='Local exported-dtype validation NMSE',
           title=f'{group["label"]}: baseline, best observed, and final export')
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, output, group['name'] + '_layer_summary', manifest, 'layer_nmse_summary',
          experiment=group['experiment'], candidate=group['candidate'])
    return summary_rows


def _diagnostic_plots(group, output, manifest):
    rows = [row for row in group['rows'] if row['record_type'] in ('step_history', 'jsonl_step')]
    for metric in OPTIONAL:
        fig, ax = None, None
        layers = []
        for layer in range(36):
            pairs = _observed([row for row in rows if row['layer'] == layer], metric)
            if not pairs:
                continue
            if fig is None:
                fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
            x, y = zip(*pairs)
            ax.scatter(x, y, s=7, alpha=0.5, label=f'Layer {layer}')
            layers.append(layer)
        if fig is None:
            continue
        ax.set(xlabel='Recorded optimizer step', ylabel=metric,
               title=f'{group["label"]}: recorded {metric}')
        ax.grid(alpha=0.2)
        _save(fig, output, group['name'] + '_' + metric, manifest, 'recorded_training_diagnostic',
              metric=metric, layers_recorded=layers, experiment=group['experiment'])


def _training_groups(root, prefix, snapshots, manifest):
    groups = {}
    for path in sorted(root.rglob('distillation_progress.json')):
        payload = snapshots.read(path)
        if isinstance(payload, list):
            groups[path.parent] = (payload, path)
    for path in sorted(root.rglob('result.json')):
        payload = snapshots.read(path)
        if isinstance(payload, dict) and isinstance(payload.get('local_distillation_layer_reports'), list):
            if path.parent not in groups:
                groups[path.parent] = (payload['local_distillation_layer_reports'], path)
    streams = {}
    for path in sorted(root.rglob('layer_*.jsonl')):
        if path.parent.name != 'training_steps':
            continue
        match = re.fullmatch(r'layer_(\d+)\.jsonl', path.name)
        if not match:
            continue
        directory = path.parent.parent
        streams.setdefault(directory, {})[int(match.group(1))] = (snapshots.lines(path), path)
        if directory not in groups:
            groups[directory] = ([], path)
    result = []
    for directory, (reports, source) in sorted(groups.items()):
        relative = str(directory.relative_to(root))
        label = f'{root.name}/{relative}' if relative != '.' else root.name
        report_by_layer = {}
        rows = []
        combined = {report.get('layer'): dict(report) for report in reports}
        if len(combined) != len(reports):
            raise ValueError(f'Duplicate layer report in {source}')
        for layer, (events, stream_path) in streams.get(directory, {}).items():
            report = combined.setdefault(layer, {'layer': layer})
            report['_jsonl_events'] = events
            report['_jsonl_source'] = str(stream_path)
            for item in events:
                event = item['event']
                if event.get('event') == 'baseline':
                    report.setdefault('baseline_val_nmse', event.get('val_nmse'))
                    report.setdefault('baseline_train_nmse', event.get('full_train_nmse'))
                if event.get('event') == 'completed':
                    for field in ('best_step', 'completed_steps', 'final_val_nmse'):
                        report.setdefault(field, event.get(field))
        for report in combined.values():
            layer = report.get('layer')
            if not isinstance(layer, int) or isinstance(layer, bool):
                manifest['missing_metrics'].append({'source': str(source), 'reason': 'Layer ID missing; report skipped'})
                continue
            if layer in report_by_layer:
                raise ValueError(f'Duplicate layer {layer} in {source}')
            report_by_layer[layer] = report
            for kind in ('history', 'step_history'):
                for sample in report.get(kind, []) or []:
                    row = {'experiment': str(root), 'candidate': relative, 'layer': layer,
                           'record_type': kind, 'source_path': str(source), 'step': sample.get('step'),
                           'observed_keys': json.dumps(sorted(sample), separators=(',', ':'))}
                    row.update({key: sample.get(key) for key in METRICS})
                    rows.append(row)
            for item in report.get('_jsonl_events', []):
                sample = item['event']
                manifest['training_events'].append({'experiment': str(root), 'candidate': relative,
                    'layer': layer, 'event': sample.get('event'), 'source_path': report['_jsonl_source'],
                    'source_line': item['line'], 'raw_event_json': json.dumps(sample, separators=(',', ':'))})
                if sample.get('event') in ('baseline', 'step'):
                    row = {'experiment': str(root), 'candidate': relative, 'layer': layer,
                           'record_type': 'jsonl_' + sample['event'], 'source_path': report['_jsonl_source'],
                           'step': sample.get('step'),
                           'observed_keys': json.dumps(sorted(sample), separators=(',', ':'))}
                    row.update({key: sample.get(key) for key in METRICS})
                    rows.append(row)
            layer_rows = [row for row in rows if row['layer'] == layer]
            missing = [metric for metric in OPTIONAL if not _observed(layer_rows, metric)]
            manifest['sampling'].append({'experiment': str(root), 'candidate': relative, 'layer': layer,
                'validation': _intervals(layer_rows, 'val_nmse'),
                'train_batch': _intervals(layer_rows, 'train_batch_nmse')})
            if missing:
                manifest['missing_metrics'].append({'experiment': str(root), 'candidate': relative,
                    'layer': layer, 'metrics': missing,
                    'reason': 'No recorded numeric samples; configured hyperparameters are not substituted'})
        missing_layers = sorted(set(range(36)) - set(report_by_layer))
        if missing_layers:
            manifest['missing_metrics'].append({'experiment': str(root), 'candidate': relative,
                                               'unrecorded_layers': missing_layers})
        if report_by_layer:
            result.append({'experiment': str(root), 'candidate': relative, 'label': label,
                           'name': prefix + '_' + _safe(relative), 'reports': report_by_layer, 'rows': rows})
    return result


def _expanded_plots(root, prefix, snapshots, output, manifest):
    comparisons, paired_rows = [], []
    expanded_results = 0
    for path in sorted(root.rglob('result.json')):
        payload = snapshots.read(path)
        if not isinstance(payload, dict) or not isinstance(payload.get('expanded_validation'), dict):
            continue
        expanded_results += 1
        label = f'{root.name}/{path.parent.relative_to(root)}'
        name = prefix + '_' + _safe(str(path.parent.relative_to(root)))
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for index, role in enumerate(('expanded_validation', 'expanded_holdout')):
            summary = payload.get(role, {})
            records = summary.get('paired_records', [])
            valid = [record for record in records if _finite(record.get('dense_minADE_m'))
                     and _finite(record.get('student_minADE_m'))]
            for field, legend, color in (('dense_minADE_m', 'Dense teacher', '#777777'),
                                         ('student_minADE_m', 'Student', '#276787')):
                values = np.sort([record[field] for record in valid])
                if len(values):
                    axes[index, 0].step(values, np.arange(1, len(values) + 1) / len(values),
                                        where='post', label=legend, color=color)
            deltas = [record['student_minADE_m'] - record['dense_minADE_m'] for record in valid]
            if deltas:
                axes[index, 1].hist(deltas, bins=min(20, max(1, int(math.sqrt(len(deltas))))),
                                    color='#c66928', alpha=0.8)
            axes[index, 1].axvline(0, color='#555555', linewidth=1, linestyle='--')
            axes[index, 0].set(title=f'{role}: ADE ECDF (n={len(valid)})', xlabel='minADE (m)', ylabel='ECDF')
            axes[index, 1].set(title=f'{role}: paired ADE differences',
                              xlabel='Student ADE - dense ADE (m)', ylabel='Clips')
            for ax in axes[index]:
                ax.grid(alpha=0.2)
            if len(valid):
                axes[index, 0].legend()
            for record in valid:
                paired_rows.append({'experiment': str(root), 'evaluation': str(path.parent.relative_to(root)),
                    'role': role, 'clip_id': record['clip_id'], 'dense_minADE_m': record['dense_minADE_m'],
                    'student_minADE_m': record['student_minADE_m'],
                    'delta_minADE_m': record['student_minADE_m'] - record['dense_minADE_m'],
                    'dense_pred_xyz_sha256': record.get('dense_pred_xyz_sha256'),
                    'student_pred_xyz_sha256': record.get('student_pred_xyz_sha256')})
            comparisons.extend(_points(root, path, payload, role, valid, snapshots, label))
        fig.suptitle(label + ': paired trajectory evaluation\n'
                     f'Status: {payload.get("status", "unrecorded")}; internal single-chunk clips', fontsize=13)
        _save(fig, output, name + '_ade_distributions', manifest, 'paired_ade_distributions', source=str(path))
    if not expanded_results:
        manifest['missing_metrics'].append({'experiment': str(root),
            'reason': 'No expanded evaluation result is recorded; no expanded ADE charts or values invented'})
    return comparisons, paired_rows


def _points(root, source, payload, role, records, snapshots, label):
    split, signature, split_path = _nearest_split(source.parent, root, snapshots)
    expected = None
    if split:
        key = role if role.startswith('expanded_') else 'validation' if role == 'validation' else 'holdout'
        expected = split.get(key)
    ids = sorted(record['clip_id'] for record in records)
    complete = expected is not None and len(ids) == len(set(ids)) and sorted(expected) == ids
    connected = complete and signature is not None and payload.get('status') == 'completed'
    result = []
    if role.startswith('expanded_'):
        metrics = [('student', payload.get('params'), [r['student_minADE_m'] for r in records]),
                   ('dense', payload.get('original_dense_params'), [r['dense_minADE_m'] for r in records])]
    else:
        metrics = [('student', payload.get('params'), [r['minADE_m'] for r in records])]
    for model, params, values in metrics:
        if _finite(params) and params > 0 and values:
            result.append({'experiment': str(root), 'source': str(source), 'label': label + ' ' + model,
                'role': role, 'model': model, 'params': params, 'mean_minADE_m': float(np.mean(values)),
                'clips': len(values), 'split_json_signature': signature, 'split_file': split_path,
                'observed_ids_signature': _sha(json.dumps(ids, separators=(',', ':')).encode()),
                'connectable': connected})
    return result


def _comparison_plot(points, output, manifest):
    if not points:
        manifest['missing_metrics'].append({'reason': 'No paired/candidate ADE records with total parameter counts'})
        return
    roles = sorted(set(point['role'] for point in points))
    fig, axes = plt.subplots(len(roles), 1, figsize=(12, 5 * len(roles)), squeeze=False, constrained_layout=True)
    for role, ax in zip(roles, axes.flat):
        selected = [point for point in points if point['role'] == role]
        # Shared dense references can appear in multiple compared experiments.
        # Plot one teacher point per identical split/value, with short labels.
        unique, seen = [], set()
        for point in selected:
            key = (point['model'], point['params'], point['mean_minADE_m'],
                   point['split_json_signature'], point['observed_ids_signature'])
            if point['model'] == 'dense' and key in seen:
                continue
            seen.add(key)
            unique.append(point)
        selected = unique
        signatures = set(point['split_json_signature'] for point in selected if point['split_json_signature'])
        midpoint = (min(p['params'] for p in selected) + max(p['params'] for p in selected)) / 2
        for index, point in enumerate(selected):
            tag = ' [split unavailable/partial]' if not point['connectable'] else ''
            if len(signatures) > 1:
                tag += ' [split ' + point['split_json_signature'][:8] + ']' if point['split_json_signature'] else ''
            ax.scatter([point['params'] / 1e9], [point['mean_minADE_m']], s=45,
                       marker='D' if point['model'] == 'dense' else 'o')
            short = re.sub(r'_20\d{6}_\d{6}$', '', Path(point['experiment']).name)
            short = short.removeprefix('mlp_').replace('_', ' ')
            label = 'Dense teacher' if point['model'] == 'dense' else short
            right = point['params'] >= midpoint
            ax.annotate(label + f' (n={point["clips"]})' + tag,
                        (point['params'] / 1e9, point['mean_minADE_m']),
                        xytext=(-8 if right else 8, 10 if index % 2 == 0 else -15),
                        ha='right' if right else 'left', textcoords='offset points', fontsize=9)
        ax.margins(x=0.08, y=0.15)
        ax.set(title=role.replace('_', ' ') + ': recorded model comparisons',
               xlabel='Total parameters (billions)', ylabel='Mean minADE (m)')
        ax.grid(alpha=0.2)
    _save(fig, output, 'parameter_ade_comparison', manifest, 'parameter_ade_comparison',
          connection_policy='Recorded points only; no interpolated tradeoff curve; split differences are labeled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', type=Path, action='append', default=[])
    parser.add_argument('--language', choices=('en', 'zh'), default='en')
    parser.add_argument('--font', type=Path, default=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'))
    args = parser.parse_args()
    global LANGUAGE, CJK_FONT
    LANGUAGE = args.language
    if LANGUAGE == 'zh':
        if not args.font.is_file():
            parser.error('Chinese output requires an existing CJK font file')
        CJK_FONT = str(args.font.resolve())
        font_manager.fontManager.addfont(CJK_FONT)
    roots = list(dict.fromkeys(path.resolve() for path in [args.experiment] + args.compare))
    if any(not root.is_dir() for root in roots):
        parser.error('Every experiment/compare path must be an existing directory')
    output = args.output.resolve() if args.output else roots[0] / 'curves'
    if output in roots:
        parser.error('output must be a separate curves directory, not an experiment root')
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    if LANGUAGE == 'zh':
        plt.rcParams['svg.fonttype'] = 'path'
    snapshots = Snapshots()
    manifest = {'schema_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'language': LANGUAGE, 'font': CJK_FONT,
                'exporter_source_sha256': _sha(Path(__file__).read_bytes()),
                'exporter_source_file': str(Path(__file__).resolve()),
                'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__,
                'python_version': sys.version, 'argv': sys.argv,
                'experiments': [str(root) for root in roots], 'charts': [], 'sampling': [], 'missing_metrics': [],
                'training_events': [],
                'metric_semantics': {'loss': 'Local cached-teacher MLP output NMSE, not end-to-end loss',
                    'train_batch_nmse': 'Observed minibatch loss, distinct from full-cache baseline/final train NMSE',
                    'best_observed_val_nmse': 'Minimum actually recorded validation NMSE (unless explicitly supplied)',
                    'interpolation': 'None; optimizer-step charts show recorded samples only',
                    'ECDF': 'Empirical step CDF; no distribution fitting or smoothing'},
                'limitations': ['Missing LR/gradient/resource samples are not reconstructed from configuration',
                                'Single-chunk internal ADE evaluation does not imply generalization or deployment safety']}
    all_rows, summaries, comparisons, pairs = [], [], [], []
    interface_present = False
    for index, root in enumerate(roots, 1):
        prefix = f'{index:02d}_' + _safe(root.name)
        for provenance in sorted(root.rglob('provenance.json')):
            snapshots.read(provenance)
        interface_present = _interface_plots(root, prefix, snapshots, output, manifest) or interface_present
        groups = _training_groups(root, prefix, snapshots, manifest)
        for group in groups:
            all_rows.extend(group['rows'])
            summaries.extend(_layer_plots(group, output, manifest))
            _diagnostic_plots(group, output, manifest)
        points, paired = _expanded_plots(root, prefix, snapshots, output, manifest)
        comparisons.extend(points)
        pairs.extend(paired)
        for path in sorted(root.rglob('result.json')):
            payload = snapshots.read(path)
            if not isinstance(payload, dict) or not isinstance(payload.get('validation'), list):
                continue
            records = [record for record in payload['validation'] if _finite(record.get('minADE_m'))]
            comparisons.extend(_points(root, path, payload, 'validation', records, snapshots,
                                        f'{root.name}/{path.parent.relative_to(root)}'))
    _csv(output / 'loss_history.csv', all_rows,
         ['experiment', 'candidate', 'layer', 'record_type', 'step', *METRICS, 'observed_keys', 'source_path'])
    _csv(output / 'training_events.csv', manifest.pop('training_events'),
         ['experiment', 'candidate', 'layer', 'event', 'source_path', 'source_line', 'raw_event_json'])
    _csv(output / 'layer_nmse_summary.csv', summaries,
         ['experiment', 'candidate', 'layer', 'baseline_val_nmse', 'best_observed_val_nmse',
          'final_val_nmse', 'best_step', 'baseline_train_nmse', 'final_train_nmse'])
    _csv(output / 'paired_ade.csv', pairs,
         ['experiment', 'evaluation', 'role', 'clip_id', 'dense_minADE_m', 'student_minADE_m',
          'delta_minADE_m', 'dense_pred_xyz_sha256', 'student_pred_xyz_sha256'])
    _csv(output / 'parameter_ade.csv', comparisons,
         ['experiment', 'source', 'label', 'role', 'model', 'params', 'mean_minADE_m', 'clips',
          'split_json_signature', 'split_file', 'observed_ids_signature', 'connectable'])
    _comparison_plot(comparisons, output, manifest)
    if not all_rows and not interface_present:
        manifest['missing_metrics'].append({'reason': 'No recorded layer history or step_history samples found'})
    manifest['inputs'] = snapshots.inputs
    manifest['snapshot_read_warnings'] = snapshots.warnings
    manifest['csv_files'] = [{'file': path.name, 'sha256': _sha(path.read_bytes())}
                             for path in sorted(output.glob('*.csv'))]
    manifest['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    (output / 'charts_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'charts': len(manifest['charts']),
                      'loss_records': len(all_rows), 'paired_ade_records': len(pairs),
                      'snapshot_warnings': len(snapshots.warnings)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
