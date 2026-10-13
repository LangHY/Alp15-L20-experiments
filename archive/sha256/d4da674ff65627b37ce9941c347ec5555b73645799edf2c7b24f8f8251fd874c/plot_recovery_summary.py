"""Compare completed recovery experiments on identical clips and dense hashes."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = [
        ('旧 8B 均匀联合蒸馏', 'mlp_stage8_joint_20261001_182631'),
        ('7.228B 非均匀局部蒸馏', 'mlp_layerwise7_20261001_220500'),
        ('7.228B 整网接口蒸馏', 'mlp_interface7_20261001_234500'),
        ('7.228B 接口与 token 蒸馏', 'mlp_interface7_logits_20261001_235500'),
        ('7.228B 接口与动作蒸馏', 'mlp_interface7_expert_20261002_001500'),
        ('7.625B 保守渐进剪枝', 'mlp_progressive76_20261002_003000'),
    ]
    models, inputs = [], []
    for label, name in sources:
        path = args.root / 'experiments' / name / 'expanded_checks/result.json'
        if not path.exists():
            continue
        data = path.read_bytes(); x = json.loads(data)
        if x['status'] != 'completed':
            continue
        models.append((label, x))
        inputs.append({'file': str(path), 'sha256': hashlib.sha256(data).hexdigest()})
    if len(models) < 2:
        raise RuntimeError('Need at least two complete paired results')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    rows = []
    for role, count in (('expanded_validation', 20), ('expanded_holdout', 69)):
        ref = models[0][1][role]
        teacher = {r['clip_id']: (r['dense_minADE_m'], r['dense_pred_xyz_sha256']) for r in ref['paired_records']}
        if len(teacher) != count:
            raise RuntimeError('Wrong frozen count')
        rows.append({'role': role, 'model': '原始教师', 'params': models[0][1]['original_dense_params'],
            'ADE_m': ref['dense_minADE_m']['mean'], 'count': count, 'missing_markers': 0, 'accepted': 'reference'})
        for label, x in models:
            group = x[role]
            identity = {r['clip_id']: (r['dense_minADE_m'], r['dense_pred_xyz_sha256']) for r in group['paired_records']}
            if teacher != identity or group['completed_pairs'] != count:
                raise RuntimeError('Clip split or dense prediction SHA differs')
            rows.append({'role': role, 'model': label, 'params': x['params'],
                'ADE_m': group['student_minADE_m']['mean'], 'count': count,
                'missing_markers': x['missing_trajectory_start_counts'][role]['student'],
                'accepted': x['next_6B_stage_gate']['pass']})
    with (out / 'same_split_recovery.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'
    font = FontProperties(fname=font_path)
    plt.rcParams['svg.fonttype'] = 'path'
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    for ax, role, title in zip(axes, ('expanded_validation', 'expanded_holdout'),
        ('固定 20 段扩展验证：决定是否接受', '固定 69 段留出：只报告')):
        points = [r for r in rows if r['role'] == role]
        labels = [f"{r['model']}（{r['params']/1e9:.3f}B）" for r in points]
        colors = ['#376b94' if r['accepted']=='reference' else '#45856a' if r['accepted'] else '#b78055' for r in points]
        ax.barh(range(len(points)), [r['ADE_m'] for r in points], color=colors, height=.6)
        ax.set_yticks(range(len(points)), labels, fontproperties=font, fontsize=10); ax.invert_yaxis()
        for index, row in enumerate(points):
            text = f"{row['ADE_m']:.3f}" + (f"；缺标记 {row['missing_markers']}" if row['missing_markers'] else '')
            ax.text(row['ADE_m']+.035, index, text, va='center', fontproperties=font, fontsize=10)
        if role == 'expanded_validation':
            limit = points[0]['ADE_m'] * 1.15
            ax.axvline(limit, linestyle='--', color='#666666', linewidth=1, label=f'验收上限 {limit:.3f} 米')
            ax.legend(prop=font, loc='lower right')
        ax.set_xlim(0, max(r['ADE_m'] for r in points)+.6)
        ax.set_title(title, fontproperties=font); ax.set_xlabel('平均轨迹 ADE（米，越低越好）', fontproperties=font)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('同一数据划分的实测对照：绿色通过验证，棕色未通过', fontproperties=font, fontsize=15)
    for suffix in ('png','svg'):
        fig.savefig(out / f'recovery_comparison_zh.{suffix}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    shutil.copy2(__file__, out/'plot_source.py')
    manifest = {'inputs': inputs, 'font': font_path, 'matplotlib': matplotlib.__version__,
        'note': 'Exact clip IDs and dense hashes verified; holdout never selects models; no interpolation',
        'outputs': [{'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir()]}
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('RECOVERY_COMPARISON_SAVED', out)


if __name__ == '__main__':
    main()
