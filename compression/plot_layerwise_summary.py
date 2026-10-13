"""Plot same-split trajectory results, including rejected diagnostic candidates."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        ('旧 8B 均匀蒸馏', 'mlp_stage8_joint_20261001_182631/expanded_checks/result.json'),
        ('新 8B 非均匀蒸馏', 'mlp_layerwise8_20261001_220500/expanded_checks/result.json'),
        ('7.228B 非均匀（未通过）', 'mlp_layerwise7_20261001_220500/expanded_checks/result.json'),
        ('旧 6B 诊断（未通过）', 'mlp_stage6_joint_20261001_184023/expanded_diagnostic_20261001_223000/result.json'),
    ]
    models, artifacts = [], []
    for label, relative in sources:
        path = args.root / 'experiments' / relative
        data = path.read_bytes()
        x = json.loads(data)
        if x['status'] != 'completed':
            raise ValueError('Every comparison must cover the complete frozen split')
        models.append((label, x))
        artifacts.append({'file': str(path), 'sha256': hashlib.sha256(data).hexdigest()})
    rows = []
    for role, expected_count in [('expanded_validation', 20), ('expanded_holdout', 69)]:
        ref = models[0][1][role]
        teacher = {r['clip_id']: (r['dense_minADE_m'], r['dense_pred_xyz_sha256']) for r in ref['paired_records']}
        if len(teacher) != expected_count:
            raise ValueError('Wrong frozen comparison count')
        rows.append({'split': role, 'model': '原始教师', 'params': models[0][1]['original_dense_params'],
                     'mean_ADE_m': ref['dense_minADE_m']['mean'], 'clips': expected_count,
                     'missing_markers': 0, 'accepted': 'reference'})
        for label, x in models:
            group = x[role]
            current = {r['clip_id']: (r['dense_minADE_m'], r['dense_pred_xyz_sha256']) for r in group['paired_records']}
            if current != teacher or group['completed_pairs'] != expected_count:
                raise ValueError('Comparison split or teacher predictions differ')
            rows.append({'split': role, 'model': label, 'params': x['params'],
                         'mean_ADE_m': group['student_minADE_m']['mean'], 'clips': expected_count,
                         'missing_markers': x['missing_trajectory_start_counts'][role]['student'],
                         'accepted': x['next_6B_stage_gate']['pass']})
    with (out / 'same_split_comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'
    font = FontProperties(fname=font_path)
    plt.rcParams['svg.fonttype'] = 'path'
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    display_names = ['原始教师', '8B 均匀', '8B 非均匀', '7.228B 非均匀', '6.266B 诊断']
    for ax, role, title in zip(axes, ['expanded_validation', 'expanded_holdout'],
                               ['固定 20 段扩展验证', '固定 69 段留出评估']):
        points = [r for r in rows if r['split'] == role]
        for i, row in enumerate(points):
            params = row['params']/1e9
            ax.scatter(params, row['mean_ADE_m'], color=f'C{i}', marker='x' if row['accepted'] is False else 'o', s=70)
            right_align = params > 10 or 7 < params < 7.8
            ax.annotate(display_names[i], (params, row['mean_ADE_m']),
                        xytext=(-8 if right_align else 8, 8), textcoords='offset points',
                        ha='right' if right_align else 'left', fontproperties=font, fontsize=10)
        ax.set_xlim(6, 11.6)
        values = [r['mean_ADE_m'] for r in points]
        ax.set_ylim(min(values)-.12, max(values)+.22)
        ax.set_title(title, fontproperties=font)
        ax.set_xlabel('总参数量（十亿）', fontproperties=font)
        ax.set_ylabel('平均轨迹 ADE（米，越低越好）', fontproperties=font)
        ax.grid(alpha=.25)
    fig.suptitle('重要性驱动剪枝：同一数据划分下的实测结果', fontproperties=font, fontsize=16)
    fig.text(.5, .015, '叉号表示未通过验证；6B 为补充诊断，保留其生成标记缺失记录', ha='center', fontproperties=font)
    fig.tight_layout(rect=[0, .04, 1, .95])
    for suffix in ['png', 'svg']:
        fig.savefig(out / f'parameter_ade_comparison_zh.{suffix}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    shutil.copy2(__file__, out / 'plot_source.py')
    manifest = {'inputs': artifacts, 'font': font_path, 'matplotlib': matplotlib.__version__,
                'note': 'Rejected candidates are reporting only; comparisons check identical clip IDs and dense prediction hashes',
                'outputs': [{'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir()]}
    (out / 'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n', encoding='utf-8')
    print('COMPARISON_SAVED', out)


if __name__ == '__main__':
    main()
