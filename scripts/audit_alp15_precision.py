#!/usr/bin/env python3
"""只读审计冻结开发集 JSON；仅标准库，不加载模型或 GT。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "docs/baselines/ALP15-PRECISION-VALIDATION-DECISION-20261009.md"
DECISION_SHA = "989b64ac1adfa3979556db82d035f71230d91132454b8eb35029eafacf0280db"
SEEDS = {42, 43, 44}
VARIANTS = {"fullprefill_10steps": 10, "actualKV_10steps": 10,
            "actualKV_6steps": 6, "actualKV_4steps": 4, "actualKV_2steps": 2}
TARGETS = {f'vlm.model.language_model.layers.{layer}.{group}.{projection}'
           for layer in range(36) for group, names in
           [('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
            ('mlp', ('gate_proj', 'up_proj', 'down_proj'))] for projection in names}
PAIR_FIELDS = ("ADE_m", "FDE_m", "initial_noise_sha256", "prefix_sha256",
               "prefix_length", "generated_token_ids", "future_start_token_id",
               "decision_id", "steps")
LIMITATIONS = ["exploratory_reused_development: 20 clips repeatedly used for selection; 3 seeds are repeated measures",
               "No GT in supplied JSON: recorded ADE/FDE cannot be independently recomputed",
               "Teacher FDE and relative teacher FDE are unavailable (null)",
               "No p values, confidence intervals, independent precision pass, safety, or generalization claim",
               "No NVIDIA original-entry, Jetson, Orin 300 ms, remote weight-byte or closed-loop acceptance",
               "No scene metadata or drive/session IDs in these result records",
               "Teacher and KV exit codes are not embedded in JSON"]


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"{label}: invalid SHA256")


def number(value, label, nonnegative=True):
    require(type(value) in (int, float), f"{label}: nonfinite or invalid numeric type")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise AuditError(f"{label}: numeric value cannot be represented as finite float") from exc
    require(finite, f"{label}: nonfinite or invalid numeric type")
    require(not nonnegative or value >= 0, f"{label}: negative value")


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label}: invalid integer")


def reject_constant(value):
    raise AuditError(f"JSON nonfinite constant: {value}")


def unique_object(items):
    result = {}
    for key, value in items:
        require(key not in result, f"JSON duplicate object key: {key}")
        result[key] = value
    return result


def parse_json(data):
    try:
        result = json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise AuditError(f"Invalid JSON: {exc}") from exc
    require(isinstance(result, dict), "JSON root must be an object")
    def finite_tree(value):
        if type(value) is float:
            require(math.isfinite(value), 'JSON contains nonfinite number')
        elif isinstance(value, dict):
            for child in value.values():
                finite_tree(child)
        elif isinstance(value, list):
            for child in value:
                finite_tree(child)
    finite_tree(result)
    return result


def quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def describe(values):
    return {"mean": statistics.mean(values), "SD": statistics.stdev(values) if len(values) > 1 else 0.,
            "median": statistics.median(values), "IQR": quantile(values, .75) - quantile(values, .25),
            "p95": quantile(values, .95), "max": max(values)}


def index_rows(rows, fields, label, teacher=False):
    require(isinstance(rows, list), f"{label}: records missing or not list")
    indexed = {}
    for n, row in enumerate(rows):
        where = f"{label} records[{n}]"
        require(isinstance(row, dict) and all(f in row for f in fields), f"{where}: missing identity fields")
        require(isinstance(row['clip_id'], str) and bool(row['clip_id']), f"{where}: invalid clip_id")
        integer(row['seed'], where + " seed")
        require(row['seed'] in SEEDS, f"{where}: unexpected seed")
        require(all(isinstance(row[f], str) for f in fields if f != 'seed'), f"{where}: invalid identity type")
        key = tuple(row[f] for f in fields)
        require(key not in indexed, f"{where}: duplicate request {key}")
        indexed[key] = row
        if not teacher:
            require(row.get('status') == 'completed', f"{where}: failed or missing status")
        number(row.get('ADE_m'), where + " ADE_m")
        trajectory = row.get('pred_xyz')
        require(isinstance(trajectory, list) and len(trajectory) == 64, f"{where}: pred_xyz must be 64x3")
        for point in trajectory:
            require(isinstance(point, list) and len(point) == 3, f"{where}: pred_xyz must be 64x3")
            for value in point:
                number(value, where + " pred_xyz", False)
        for field in ('prefix_sha256', 'initial_noise_sha256', 'pred_xyz_sha256'):
            digest(row.get(field), where + " " + field)
        if 'prefix_length' in row:
            integer(row['prefix_length'], where + ' prefix_length', 1)
        if 'generated_tokens' in row:
            integer(row['generated_tokens'], where + ' generated_tokens', 1)
        for field in ('FDE_m', 'e2e_ms', 'teacher_ADE_m', 'original_awq_ADE_m'):
            if not teacher or field in row:
                if field in row or field in ('FDE_m', 'e2e_ms', 'teacher_ADE_m'):
                    number(row.get(field), where + " " + field)
        for field in ('stage_ms', 'separate_io_ms', 'separate_io_and_validation_ms'):
            if field in row:
                require(isinstance(row[field], dict), where + ' ' + field + ': expected timings object')
                for name, value in row[field].items():
                    number(value, where + ' ' + name)
        if 'separate_metric_and_math_audit_ms' in row:
            number(row['separate_metric_and_math_audit_ms'], where + ' metric timing')
    return indexed


def check_coverage(index, clips, prefix, label):
    expected = {tuple(prefix) + (clip, seed) for clip in clips for seed in SEEDS}
    require(set(index) == expected, f"{label}: missing/extra clip or seed coverage ({len(index)} rows)")


def hash_map(value, label):
    require(isinstance(value, dict) and bool(value), f"{label}: missing hash map")
    for key, val in value.items():
        require(isinstance(key, str) and bool(key), f"{label}: invalid path")
        digest(val, label + " " + key)


def protected(obj, label):
    for f in ('protected_sha256_before', 'protected_sha256_after'):
        digest(obj.get(f), label + " " + f)
    require(obj['protected_sha256_before'] == obj['protected_sha256_after'], f"{label}: protected hash changed")


def dispatch(obj, indexed, enabled, label):
    install = obj.get('hybrid_install', {})
    require(install.get('coverage') == '252/252' and install.get('enabled') is enabled
            and install.get('persistent_dense') is False, f"{label}: invalid hybrid install")
    targets = install.get('target_names')
    require(isinstance(targets, list) and all(isinstance(t, str) for t in targets)
            and len(targets) == len(set(targets)) == 252, f"{label}: target coverage")
    require(set(targets) == TARGETS, f"{label}: wrong frozen target names")
    integer(install.get('packed_bytes_before'), label + " packed_bytes_before")
    require(install.get('packed_bytes_after') == install['packed_bytes_before'], f"{label}: packed bytes changed")
    for key, row in indexed.items():
        d = row.get('dispatch', {})
        modules, totals = d.get('modules'), d.get('totals')
        require(type(d.get('module_count')) is int and d['module_count'] == 252
                and isinstance(modules, dict) and set(modules) == set(targets), f"{label} {key}: dispatch coverage")
        require(isinstance(totals, dict) and set(totals) == {'m1_fused', 'm_gt1_reference', 'disabled_reference'}, f"{label}: dispatch totals fields")
        sums = dict.fromkeys(totals, 0)
        for module, counts in modules.items():
            require(isinstance(counts, dict) and set(counts) == set(totals), f"{label}: dispatch counts fields")
            for field, count in counts.items():
                integer(count, f"{label} {module} {field}")
                sums[field] += count
            require(counts['m1_fused'] > 0 if enabled else counts['m1_fused'] == counts['m_gt1_reference'] == 0 and counts['disabled_reference'] > 0,
                    f"{label}: dispatch path incomplete")
        for field, count in totals.items():
            integer(count, f"{label} totals {field}")
        require(totals == sums, f"{label}: dispatch totals mismatch")
    return set(targets)


def saved_gates(rows, clips, label, strict=False):
    require(isinstance(rows, list), f"{label}: missing gates")
    index = {}
    for row in rows:
        require(isinstance(row, dict) and 'clip_id' in row and 'seed' in row, f"{label}: missing gate identity")
        key = (row['clip_id'], row['seed'])
        require(key not in index, f"{label}: duplicate gate")
        index[key] = row
        for field in ('trajectory_nmse', 'absolute_ADE_delta_m', 'max_abs_xyz_delta_m'):
            number(row.get(field), label + " " + field)
            require(row[field] == 0, f"{label}: saved exact gate contradicts arrays")
        flags = ['noise_paired', 'passed']
        if strict:
            flags += ['prefix_sha256_paired', 'generated_token_ids_paired', 'future_start_token_id_paired', 'prefix_length_paired', 'request_identity_paired']
        require(all(row.get(field) is True for field in flags), f"{label}: gate false or invalid boolean")
    check_coverage(index, clips, (), label)
    return index


def audit(data, hashes):
    teacher, kv, reference, hybrid = (data[n] for n in ('teacher', 'kv', 'reference', 'hybrid'))
    require(teacher.get('status') == 'completed', "teacher: top-level status")
    ti = index_rows(teacher.get('records'), ('model', 'prefix', 'clip_id', 'seed'), 'teacher', True)
    groups = {(key[0], key[1]) for key in ti}
    require(groups == {('dense', 'teacher'), ('student8', 'teacher'), ('student8', 'student8'), ('dense', 'student8')}, "teacher: wrong teacher groups")
    clips = sorted({key[2] for key in ti if key[:2] == ('dense', 'teacher')})
    require(len(clips) == 20, "teacher: must contain 20 clips")
    for group in groups:
        check_coverage({key: row for key, row in ti.items() if key[:2] == group}, clips, group, "teacher " + str(group))
    teachers = {(key[2], key[3]): row for key, row in ti.items() if key[:2] == ('dense', 'teacher')}
    require(kv.get('status') == 'completed_kv_steps_development_probe', "kv: top-level status")
    ki = index_rows(kv.get('records'), ('variant', 'clip_id', 'seed'), 'kv')
    require({key[0] for key in ki} == set(VARIANTS), "kv: variant groups")
    candidates = {'teacher_dense_teacher': teachers}
    for variant, steps in VARIANTS.items():
        subset = {key: row for key, row in ki.items() if key[0] == variant}
        check_coverage(subset, clips, (variant,), variant)
        require(all(type(row.get('steps')) is int and row['steps'] == steps for row in subset.values()), variant + ": steps")
        candidates[variant] = {(key[1], key[2]): row for key, row in subset.items()}
    for label, obj, variant in [('reference', reference, 'reference'), ('hybrid', hybrid, 'hybrid_m1')]:
        require(obj.get('status') == 'completed_awq_hybrid_development_eval' and type(obj.get('exit_code')) is int and obj['exit_code'] == 0, label + ": status/exit code")
        require(obj.get('request_route') == 'fullprefill' and obj.get('provenance_version') == 3, label + ": route/provenance")
        indexed = index_rows(obj.get('records'), ('variant', 'clip_id', 'seed'), label)
        check_coverage(indexed, clips, (variant,), label)
        for row in indexed.values():
            require(type(row.get('steps')) is int and row['steps'] == 10, label + ": steps")
            integer(row.get('prefix_length'), label + " prefix_length", 1)
            integer(row.get('future_start_token_id'), label + " future_start_token_id")
            tokens = row.get('generated_token_ids')
            require(isinstance(tokens, list) and bool(tokens), label + ": missing tokens")
            for token in tokens:
                integer(token, label + " generated token")
            require(tokens[-1] == row['future_start_token_id'], label + ": generation boundary")
            require(isinstance(row.get('decision_id'), str) and row['decision_id'] == obj.get('decision_id'), label + ": decision identity")
        candidates[label] = {(key[1], key[2]): row for key, row in indexed.items()}
    identities = ('source_commit', 'checkpoint_revision', 'dataset_revision', 'text_seed', 'trajectory_seeds')
    for label, obj in data.items():
        protected(obj, label)
        require(type(obj.get('text_seed')) is int and obj['text_seed'] == 42
                and obj.get('trajectory_seeds') == [42, 43, 44], label + ": seeds")
        for field in identities:
            require(field in obj and obj[field] == teacher[field], label + ": source/seed identity " + field)
        if label != 'teacher':
            hash_map(obj.get('input_sha256'), label + " input_sha256")
            binding = [(path, val) for path, val in obj['input_sha256'].items() if path.endswith('/controlled8_expanded20_20261002_125000/result.json')]
            require(len(binding) == 1 and binding[0][1] == hashes['teacher'], label + ": teacher source SHA binding")
    for field in ('code_sha256', 'runtime_source_sha256', 'input_sha256'):
        hash_map(reference.get(field), "reference " + field)
        hash_map(hybrid.get(field), "hybrid " + field)
        require(reference[field] == hybrid[field], "hybrid/reference: source hashes differ " + field)
    require(len({reference[f] for f in ('protected_sha256_before', 'protected_sha256_after')} |
                {hybrid[f] for f in ('protected_sha256_before', 'protected_sha256_after')}) == 1, "hybrid/reference: protected hashes differ")
    require(reference.get('virtual_import_sources') == hybrid.get('virtual_import_sources') and bool(reference.get('virtual_import_sources')), "virtual source identity")
    rd, hd = reference.get('dynamic_import_sources'), hybrid.get('dynamic_import_sources')
    require(isinstance(rd, dict) and isinstance(hd, dict) and set(rd) == set(hd) and bool(rd), "dynamic source identities")
    for name in rd:
        for field in ('stable_identity_sha256', 'source_sha256'):
            digest(rd[name].get(field), "dynamic " + field)
            require(rd[name][field] == hd[name].get(field), "dynamic source hash differs")
        for field in ('generators', 'generation_mode', 'generator_entry', 'torch_version', 'snapshot_policy'):
            require(rd[name].get(field) == hd[name].get(field), "dynamic generator identity differs")
    paired = hybrid.get('paired_reference', {})
    require(paired.get('sha256') == hashes['reference'] and isinstance(paired.get('result'), str), "hybrid: reference SHA binding")
    require(hybrid.get('gate_provenance', {}).get(paired['result']) == hashes['reference'], "hybrid: reference gate provenance")
    require(dispatch(reference, candidates['reference'], False, 'reference') == dispatch(hybrid, candidates['hybrid'], True, 'hybrid'), "hybrid/reference targets differ")
    pairs = []
    for key in teachers:
        r, h = candidates['reference'][key], candidates['hybrid'][key]
        results = {field: r[field] == h[field] for field in PAIR_FIELDS}
        delta = max(abs(a - b) for rp, hp in zip(r['pred_xyz'], h['pred_xyz']) for a, b in zip(rp, hp))
        exact = r['pred_xyz'] == h['pred_xyz']
        pairs.append({'clip_id': key[0], 'seed': key[1], 'coordinates_exact': exact, 'max_abs_xyz_delta_m': delta, 'fields': results})
        require(exact and all(results.values()), f"hybrid/reference {key}: pair mismatch")
    strict = saved_gates(hybrid.get('strict_pair_gates'), clips, 'hybrid strict gates', True)
    require(reference.get('strict_pair_gates') == [], "reference: unexpected strict gates")
    for key, row in strict.items():
        require(row.get('decision_id') == candidates['hybrid'][key]['decision_id'], "strict gate decision identity")
    saved_gates(hybrid.get('original_helper_gates'), clips, 'hybrid original gates')
    saved_gates(reference.get('original_helper_gates'), clips, 'reference original gates')
    saved_gates(kv.get('original_helper_gates'), clips, 'kv original gates')
    numerical = kv.get('numerical_gates')
    require(isinstance(numerical, list), 'kv: missing numerical gates')
    numerical_index = {}
    for gate in numerical:
        require(isinstance(gate, dict) and 'clip_id' in gate and 'seed' in gate, 'kv: numerical gate identity')
        key = (gate['clip_id'], gate['seed'])
        require(key not in numerical_index, 'kv: duplicate numerical gate')
        numerical_index[key] = gate
        for field in ('trajectory_nmse', 'absolute_ADE_delta_m', 'max_abs_xyz_delta_m'):
            number(gate.get(field), 'kv numerical ' + field)
        for field in ('noise_paired', 'passed', 'prefix_paired', 'numerical_gate_passed', 'approved_as_new_candidate'):
            require(type(gate.get(field)) is bool, 'kv numerical boolean ' + field)
        require(key in candidates['fullprefill_10steps'], 'kv: numerical gate extra key')
        a, b = candidates['fullprefill_10steps'][key], candidates['actualKV_10steps'][key]
        flat_a = [v for point in a['pred_xyz'] for v in point]
        flat_b = [v for point in b['pred_xyz'] for v in point]
        # Python 3.12改变float内置sum；fsum稳定复现冻结的float NMSE，不放宽模型门禁。
        nmse = math.fsum((x-y)**2 for x, y in zip(flat_a, flat_b)) / max(math.fsum(x*x for x in flat_a), 1e-12)
        ade_delta = abs(a['ADE_m'] - b['ADE_m'])
        max_delta = max(abs(x-y) for x, y in zip(flat_a, flat_b))
        require(gate['trajectory_nmse'] == nmse and gate['absolute_ADE_delta_m'] == ade_delta
                and gate['max_abs_xyz_delta_m'] == max_delta, 'kv numerical saved values disagree with arrays')
        noise, prefix = a['initial_noise_sha256'] == b['initial_noise_sha256'], a['prefix_sha256'] == b['prefix_sha256']
        passed = noise and prefix and nmse <= 1e-3 and ade_delta <= 1e-4
        require(gate['noise_paired'] == noise and gate['prefix_paired'] == prefix
                and gate['passed'] == gate['numerical_gate_passed'] == passed, 'kv numerical saved flags disagree with arrays')
    check_coverage(numerical_index, clips, (), 'kv numerical gates')
    for label, rows in candidates.items():
        if label == 'teacher_dense_teacher':
            continue
        for key, row in rows.items():
            require(row['teacher_ADE_m'] == teachers[key]['ADE_m'], f"{label} {key}: teacher ADE binding")
            require(row['initial_noise_sha256'] == teachers[key]['initial_noise_sha256'], f"{label} {key}: teacher noise binding")
    # 原 helper 的 reference 精确主张同时用实际旧 fullprefill10 数组交叉核验。
    for key, row in candidates['reference'].items():
        old = candidates['fullprefill_10steps'][key]
        require(row['pred_xyz'] == old['pred_xyz'] and row['ADE_m'] == old['ADE_m']
                and row['FDE_m'] == old['FDE_m'] and row['initial_noise_sha256'] == old['initial_noise_sha256'], "reference original helper array crosscheck")
    per_request, per_clip, summaries = [], [], {}
    teacher_means = {clip: statistics.mean(teachers[clip, seed]['ADE_m'] for seed in SEEDS) for clip in clips}
    teacher_overall = statistics.mean(teacher_means.values())
    require(teacher_overall > 0, "overall teacher ADE is zero: ratio undefined")
    for label, rows in candidates.items():
        is_teacher = label == 'teacher_dense_teacher'
        clip_rows = []
        for clip in clips:
            rs = [rows[clip, seed] for seed in sorted(SEEDS)]
            ades = [row['ADE_m'] for row in rs]
            student, baseline = statistics.mean(ades), teacher_means[clip]
            c = {'candidate': label, 'clip_id': clip, 'seeds': '42/43/44', 'teacher_ADE_m': baseline,
                 'student_ADE_m': student, 'ADE_delta_m': student - baseline,
                 'ADE_ratio': student / baseline - 1 if baseline else None,
                 'ratio_missing_reason': '' if baseline else 'teacher clip ADE is zero',
                 'seed_ADE_min_m': min(ades), 'seed_ADE_max_m': max(ades), 'seed_ADE_SD_m': statistics.stdev(ades),
                 'student_FDE_m': None if is_teacher else statistics.mean(row['FDE_m'] for row in rs),
                 'teacher_FDE_m': None, 'failed_requests': 0, 'request_denominator': 3, 'paired': True}
            clip_rows.append(c)
            for seed, row in zip(sorted(SEEDS), rs):
                per_request.append({'candidate': label, 'clip_id': clip, 'seed': seed, 'ADE_m': row['ADE_m'],
                                    'teacher_ADE_m': teachers[clip, seed]['ADE_m'], 'FDE_m': None if is_teacher else row['FDE_m'],
                                    'teacher_FDE_m': None, 'e2e_ms': row.get('e2e_ms'), 'status': 'completed' if not is_teacher else 'not_embedded_in_json',
                                    'teacher_ADE_bound': True, 'noise_paired': True, 'trajectory_shape': '64x3'})
        per_clip.extend(clip_rows)
        stats = describe([c['student_ADE_m'] for c in clip_rows])
        summaries[label] = {'clip_count': 20, 'request_count': 60, 'ADE_clip': stats,
                            'teacher_mean_ADE_m': teacher_overall, 'relative_teacher_ADE_loss': stats['mean'] / teacher_overall - 1,
                            'development_point_estimate_within_15pct': stats['mean'] / teacher_overall - 1 <= .15,
                            'clips_improved': sum(c['ADE_delta_m'] < 0 for c in clip_rows), 'clips_tied': sum(c['ADE_delta_m'] == 0 for c in clip_rows),
                            'clips_degraded': sum(c['ADE_delta_m'] > 0 for c in clip_rows), 'clips_over_15pct': sum(c['ADE_ratio'] is not None and c['ADE_ratio'] > .15 for c in clip_rows),
                            'FDE_clip': None if is_teacher else describe([c['student_FDE_m'] for c in clip_rows]),
                            'teacher_FDE': None, 'relative_teacher_FDE': None, 'FDE_missing_reason': 'teacher JSON has no GT or FDE',
                            'ADE_p95_delta_m': stats['p95'] - describe(list(teacher_means.values()))['p95'],
                            'ADE_max_delta_m': stats['max'] - max(teacher_means.values()),
                            'failed_requests': 0, 'request_failure_denominator': 60, 'request_failure_rate': 0.,
                            'failed_clips': 0, 'clip_failure_denominator': 20, 'clip_failure_rate': 0.}
    return {'status': 'completed_development_audit', 'integrity_passed': True,
            'analysis_scope': 'exploratory_reused_development', 'independent_precision_pass': None,
            'Orin_gate_passed': None, 'diagnostics': [], 'pairs': pairs, 'pair_exact_count': 60,
            'diagnostic_counts': {'nonfinite': 0, 'invalid_generation_boundary': 0, 'missing_requests': 0,
                                  'duplicate_requests': 0, 'failed_record_status': 0, 'unpaired_requests': 0},
            'teacher_group_counts': {f'{m}/{p}': 60 for m, p in sorted(groups)},
            'teacher_excluded_groups': ['student8/teacher', 'student8/student8', 'dense/student8'],
            'exit_code_evidence': {'teacher': 'not_embedded_in_json', 'kv': 'not_embedded_in_json', 'reference': 0, 'hybrid': 0},
            'kv_numerical_gates': numerical,
            'summaries': summaries, 'per_request': per_request, 'per_clip': per_clip, 'limitations': LIMITATIONS}


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows, fields):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report_text(result, manifest):
    lines = ['# 已有开发证据离线精度审计', '', f"状态：{result['status']}；范围：exploratory_reused_development。", '',
             '整体比例 = mean_clip(mean_seed student ADE) / mean_clip(mean_seed teacher ADE) − 1；clip 等权。',
             '15%仅是整体点估计描述门限；逐clip超过15%作为诊断，不替换主门禁。', '',
             f"输入/完整性诊断：{json.dumps(result.get('diagnostics', []), ensure_ascii=False)}", '',
             '|候选|mean ADE|SD|median|IQR|p95|max|整体相对损失|退化clip|超过15%clip|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, summary in result.get('summaries', {}).items():
        stats = summary['ADE_clip']
        lines.append('|' + '|'.join([name] + [f"{stats[k]:.10f}" for k in ('mean', 'SD', 'median', 'IQR', 'p95', 'max')] +
                                  [f"{summary['relative_teacher_ADE_loss']:.10%}", str(summary['clips_degraded']), str(summary['clips_over_15pct'])]) + '|')
    lines += ['', ('各候选请求失败率0/60、clip失败率0/20。' if result['integrity_passed'] else
                    '完整性失败；不计算成功样本均值或失败率，已解析的全部请求留在per_request.csv，待审计记录不能当作模型成功。'),
              f"hybrid/reference 实际坐标 exact：{result.get('pair_exact_count', 0)}/60。", '',
              '所有请求及clip见CSV；下面列出所有ADE退化clip，未删除尾部或异常样本。', '', '|候选|clip|ADE绝对增量 m|相对增量|', '|---|---|---:|---:|']
    for row in result.get('per_clip', []):
        if row['ADE_delta_m'] > 0:
            ratio = 'null' if row['ADE_ratio'] is None else f"{row['ADE_ratio']:.10%}"
            lines.append(f"|{row['candidate']}|{row['clip_id']}|{row['ADE_delta_m']:.10f}|{ratio}|")
    lines += ['', 'FDE仅为学生既有字段的clip描述；教师FDE和相对FDE为null。完整均值/中位数/p95/max以及教师ADE尾部差见audit.json。', '', '限制：']
    lines[-1:] = ['|候选|学生FDE mean|median|p95|max|ADE p95相比教师增量|ADE max相比教师增量|',
                  '|---|---:|---:|---:|---:|---:|---:|']
    for name, summary in result.get('summaries', {}).items():
        fde = summary['FDE_clip']
        values = ['null'] * 4 if fde is None else [f"{fde[k]:.10f}" for k in ('mean', 'median', 'p95', 'max')]
        lines.append('|' + '|'.join([name] + values + [f"{summary[k]:.10f}" for k in ('ADE_p95_delta_m', 'ADE_max_delta_m')]) + '|')
    lines += ['', '限制：']
    lines.extend('- ' + s for s in LIMITATIONS)
    lines += ['', '来源：', '```json', json.dumps(manifest, ensure_ascii=False, indent=2), '```', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher', 'kv', 'reference', 'hybrid'):
        parser.add_argument('--' + name, required=True, type=Path)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    # 输出目录拒绝覆盖在任何输入处理前检查；不提供force或自动续写。
    if args.output.exists():
        parser.error('output already exists; refusing overwrite')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'argv': [sys.executable, str(Path(__file__).resolve())] + (list(argv) if argv is not None else sys.argv[1:]),
                'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'platform': platform.platform(),
                'script_sha256': sha(Path(__file__).read_bytes()), 'decision_path': str(DECISION), 'inputs': {}}
    data, hashes = {}, {}
    try:
        manifest['decision_sha256'] = sha(DECISION.read_bytes())
        require(manifest['decision_sha256'] == DECISION_SHA, 'frozen decision SHA mismatch')
        for name in ('teacher', 'kv', 'reference', 'hybrid'):
            path, expected = getattr(args, name), getattr(args, name + '_sha256')
            info = {'path': str(path.resolve()), 'expected_sha256': expected}
            manifest['inputs'][name] = info
            raw = path.read_bytes()
            info['sha256_before'] = sha(raw)
            digest(expected, name + ' expected SHA')
            require(info['sha256_before'] == expected, name + ': input SHA mismatch')
            data[name] = parse_json(raw)
            info['sha256_after_parse'] = sha(path.read_bytes())
            require(info['sha256_after_parse'] == expected, name + ': input changed during parsing')
            records = data[name].get('records')
            info['record_container_type'] = type(records).__name__ if 'records' in data[name] else 'missing'
            info['record_count'] = len(records) if isinstance(records, list) else None
            require(isinstance(records, list), name + ': records missing or not list')
            for index, row in enumerate(records):
                require(isinstance(row, dict), f'{name}: records[{index}] must be an object')
            hashes[name] = expected
        result = audit(data, hashes)
    except (AuditError, OSError, KeyError, TypeError) as exc:
        inventory = []
        # 失败记录的输入身份和现有状态留存；不从无效记录计算达标结论。
        for label, obj in data.items():
            if not isinstance(obj, dict) or not isinstance(obj.get('records'), list):
                continue
            for index, row in enumerate(obj['records']):
                if isinstance(row, dict):
                    inventory.append({'input': label, 'record_index': index,
                                      'variant': row.get('variant'), 'model': row.get('model'), 'prefix': row.get('prefix'),
                                      'clip_id': row.get('clip_id'), 'seed': str(row.get('seed')),
                                      'observed_status': row.get('status', 'not_embedded_in_json'),
                                      'audit_validated': False, 'diagnostic': str(exc)})
        result = {'status': 'failed_development_audit', 'integrity_passed': False, 'analysis_scope': 'exploratory_reused_development',
                  'independent_precision_pass': None, 'Orin_gate_passed': None, 'diagnostics': [str(exc)],
                  'per_request': inventory, 'per_clip': [], 'summaries': {}, 'limitations': LIMITATIONS}
    for name, info in manifest['inputs'].items():
        try:
            info['sha256_after_audit'] = sha(getattr(args, name).read_bytes())
            if info['sha256_after_audit'] != info.get('sha256_before'):
                result['diagnostics'].append(name + ': input changed during audit')
                result.update(status='failed_development_audit', integrity_passed=False, summaries={})
        except OSError as exc:
            result['diagnostics'].append(str(exc))
            result.update(status='failed_development_audit', integrity_passed=False, summaries={})
    manifest['ended_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['exit_code'] = 0 if result['integrity_passed'] else 1
    result['provenance'] = manifest
    write_json(args.output / 'audit.json', result)
    request_fields = list(result['per_request'][0]) if result['per_request'] else ['candidate', 'clip_id', 'seed', 'status']
    clip_fields = list(result['per_clip'][0]) if result['per_clip'] else ['candidate', 'clip_id', 'status']
    write_csv(args.output / 'per_request.csv', result['per_request'], request_fields)
    write_csv(args.output / 'per_clip.csv', result['per_clip'], clip_fields)
    with (args.output / 'report.md').open('x', encoding='utf-8') as stream:
        stream.write(report_text(result, manifest))
    with (args.output / 'run.log').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps({'argv': manifest['argv'], 'status': result['status'], 'diagnostics': result['diagnostics'], 'exit_code': manifest['exit_code']}, ensure_ascii=False, indent=2) + '\n')
    manifest['output_sha256'] = {p.name: sha(p.read_bytes()) for p in args.output.iterdir() if p.is_file()}
    write_json(args.output / 'input_manifest.json', manifest)
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()), 'exit_code': manifest['exit_code'], 'diagnostics': result['diagnostics']}, ensure_ascii=False))
    return manifest['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
