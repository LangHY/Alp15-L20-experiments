# 精度优先：未见数据来源核查与用户下载命令

日期：2026-10-09。文件所有者：`precision_dataset_domestic_source_20261009`。范围仅为来源核查、下载规划与命令；本轮没有下载视频 ZIP、模型或全局 parquet，没有连接服务器或安装依赖。本文件不构成独立测试集验收。

## 1. 来源核查结论

权威数据固定为 [NVIDIA PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/tree/33f9bf447ed3bcb7d545ce13f4226f824214fafb)，revision 为 `33f9bf447ed3bcb7d545ce13f4226f824214fafb`。不要使用未固定的 `main`，也不要把 NuRec USDZ 重建场景当作四路摄像头与 egomotion 原数据。

| 来源 | 本轮实际读取的一手证据 | 可以得出的结论 |
| --- | --- | --- |
| [魔搭同名数据仓](https://www.modelscope.cn/datasets/nv-community/PhysicalAI-Autonomous-Vehicles) | [固定提交树 API](https://modelscope.cn/api/v1/datasets/nv-community/PhysicalAI-Autonomous-Vehicles/repo/tree?Revision=391de47ee091fbe53a1f4e9f0d26cd6f1e22313c&Recursive=false&PageSize=100) 返回 Code 200、顶层 TotalCount 4：`calibration/`、`.gitattributes`、`LICENSE.pdf`、`README.md`。该固定提交为 `391de47ee091fbe53a1f4e9f0d26cd6f1e22313c`。 | 缺 `camera/`、`labels/`、`metadata/`、`clip_index.parquet`、`features.csv`；无法提供本任务所需五路分块。不能认定与官方冻结 revision 相同。 |
| [HF-Mirror](https://hf-mirror.com) | [固定 revision API](https://hf-mirror.com/api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/revision/33f9bf447ed3bcb7d545ce13f4226f824214fafb) 返回准确 sha、`gated=auto`、70775 个 siblings；[顶层树 API](https://hf-mirror.com/api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/tree/33f9bf447ed3bcb7d545ce13f4226f824214fafb?recursive=false&expand=false) 与官方树的 path、Git oid、size 相同；metadata 子树也一致。 | 只证明索引指向目标 revision。匿名 API 的 LFS SHA256 被遮蔽为星号；未进行文件内容对照，未证明国内文件传输可用。 |
| HF-Mirror 实际文件入口 | 对冻结 `features.csv` 及容量示例 `camera_front_wide_120fov.chunk_0001.zip` 执行匿名 `curl -I`，均 HTTP 308，Location 为相同路径的 `https://huggingface.co/...`。跟随 `features.csv` 后 HTTP 401、`x-error-code: GatedRepo`。 | 当前访问路径返回官方站，不能承诺国内大文件节点或免授权下载。没有把既有 token 发送给镜像。 |
| NVIDIA 官方 HF | 冻结树公开可读；通过本机既有 HF 认证只读取 32853 字节 README，HTTP 200；官方 `HfApi.get_paths_info` 返回 metadata 与示例五 ZIP 的 size、LFS SHA256。 | 可以制定固定来源、固定校验信息的用户命令；本轮未实际下载这些文件，不能宣称下载或数据独立性验收通过。 |

[镜像站自身说明](https://hf-mirror.com)只称其为公益镜像，并说明 gated 仓仍须先在 HF 官网获取许可。该站不是 NVIDIA 官方数据发行源。名称、数据卡复制、revision 字符串或文件名一致，都不足以证明镜像文件字节一致。若以后发现完整国内源，应逐文件对照官方固定 revision 的 size 和 SHA256，分别记录 `authority_source/revision` 与 `transport_source/revision`；一致性确认前不替换来源。

现有 `scripts/prepare_alpamayo15_offline_bundle.sh` 对 `--dataset-source modelscope` 的拒绝仍符合本轮证据。`scripts/prefetch_physical_ai_clip.py` 固定旧 clip `030c760c-ae38-49aa-9ad8-f5650a545d26`、chunk 3119，不能直接用来获取新 holdout，也不要运行整包准备脚本导致额外模型下载。本轮没有修改它们。

## 2. 固定 API、输入及筛选边界

使用 [NVlabs/physical_ai_av 固定源码](https://github.com/NVlabs/physical_ai_av/tree/59d578b53ef186ffc3f1de1d1b2598717a982d64) 的 `physical_ai_av==0.2.0` 顶层 `PhysicalAIAVDatasetInterface`。本机既有 `/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3` 的版本已只读核对为 0.2.0；其 `dataset.py` 与[固定官方源码](https://github.com/NVlabs/physical_ai_av/blob/59d578b53ef186ffc3f1de1d1b2598717a982d64/src/physical_ai_av/dataset.py) SHA256 同为 `b35f5f1bbb42ba92c83c835f7525c33e8ee915eb5ef071ac3dd972d91b0d9560`。

官方构造器只获取四项全局元数据。`get_clip_chunk(clip_id)` 解析真实分块；`interface.features.get_chunk_feature_filename(chunk_id, feature)` 解析文件路径。仅使用以下五项，不用 `.CAMERA.ALL`、`features=None` 或整个仓库快照下载：

- `LABELS.EGOMOTION`（保留当前原 `egomotion`，不替换成 `.offline`）。
- `CAMERA.CAMERA_CROSS_LEFT_120FOV`。
- `CAMERA.CAMERA_FRONT_WIDE_120FOV`。
- `CAMERA.CAMERA_CROSS_RIGHT_120FOV`。
- `CAMERA.CAMERA_FRONT_TELE_30FOV`。

全部评测输入固定 `t0_us=5100000`。分块最多约 100 clips；下载一个 clip 通常也需整块五 ZIP，不能按 clip 数线性估算下载量。原 `chunk0000` 的 80 training/20 monitor 和旧 `chunk3119` 开发、校准、选择数据全部排除。所有训练、蒸馏、量化校准、监控、smoke、开发评测及候选选择历史必须汇入 usage ledger；以下元数据候选列表只排除了已知两个分块，不代表已经完成历史排重。

已有只读盘点证明 `data_collection` 有 `country/month/hour_of_day/platform_class/radar_config/clip_id`，没有 drive/session 字段。可以按这些真实字段分层；不得猜测或从 UUID、chunk 编号构造行程标识。新 chunk/clip 与已用列表无交集只支持相应层级的排重，不能证明 drive-disjoint，也不能据此命名为官方验证集。官方 README 的 reasoning 子集 Train 表不等于原数据的通用官方验证划分。

## 3. 下载规模：来自固定 revision 的公开 size

[官方顶层树](https://huggingface.co/api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/tree/33f9bf447ed3bcb7d545ce13f4226f824214fafb?recursive=false&expand=false)与[metadata 子树](https://huggingface.co/api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/tree/33f9bf447ed3bcb7d545ce13f4226f824214fafb/metadata?recursive=false&expand=false)给出：

| 全局元数据 | bytes | 官方 SHA256 / Git blob |
| --- | ---: | --- |
| `features.csv` | 7076 | Git blob `65de094a4037902a40bd42de7b17082029b78fe2` |
| `clip_index.parquet` | 11132300 | SHA256 `9630a3de5ee057ded194deaa94d3b9b42af6f18c301785706e50141314a6935c` |
| `metadata/feature_presence.parquet` | 11209157 | SHA256 `a9946feb269124e3a7cbfe22d97669585735134f634c2ca97332e7abc4314790` |
| `metadata/data_collection.parquet` | 11339400 | SHA256 `9624750cadabf8028264cc1404e75e81777654be9def06b00e35c9f834b173d5` |
| 合计 | **33687933** | 约 33.69 MB / 32.13 MiB，不含日志及候选 CSV |

仅为说明整块容量，查询了未选定的 **chunk0001** 文件信息，未下载，也未把它定为测试集。官方 `get_paths_info` 与[公开 camera 树](https://huggingface.co/api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/tree/33f9bf447ed3bcb7d545ce13f4226f824214fafb/camera/camera_front_wide_120fov?recursive=false&expand=false&limit=10)返回：

| chunk0001 特征，仅容量例 | bytes |
| --- | ---: |
| egomotion | 39117630 |
| cross left 120 | 2374247875 |
| front wide 120 | 2012955402 |
| cross right 120 | 2530617068 |
| front tele 30 | 1692437461 |
| 五 ZIP 合计 | **8649375436**（约 8.65 GB / 8.06 GiB） |

其他分块大小可能不同。正式规模必须先筛选元数据、冻结名单、按真实分块去重，再从官方固定 revision 查询五路 size 求和。缓存占用与解压空间分开计算；下列命令只保留 ZIP 和 HF 缓存，不展开视频，不复制第二份整块数据。

## 4. 用户命令 A：只获取元数据，生成候选表

这是官方来源备用命令，**不是已核验可用的国内文件下载命令**。需已有 HF 许可与登录状态；不在命令行写 token，不代替用户接受协议。使用本机现有 venv，不升级依赖。输出目录必须不存在；失败时保留目录和缓存，先查看错误再选择新的输出目录。命令只下载上表四项约 33.69 MB 元数据，不下载 ZIP。

```sh
env HF_ENDPOINT=https://huggingface.co \
  /Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B - <<'PY'
import hashlib, importlib.metadata, json, pathlib
import pandas as pd
from huggingface_hub import HfApi
import physical_ai_av as av

REV = '33f9bf447ed3bcb7d545ce13f4226f824214fafb'
OUT = pathlib.Path('/Users/USER/Downloads/alp15-precision-data-20261009')
assert importlib.metadata.version('physical-ai-av') == '0.2.0'
OUT.mkdir(mode=0o700, exist_ok=False)
api = HfApi(endpoint='https://huggingface.co')
assert api.dataset_info('nvidia/PhysicalAI-Autonomous-Vehicles', revision=REV).sha == REV
ds = av.PhysicalAIAVDatasetInterface(revision=REV, cache_dir=str(OUT/'hf-cache'))
specs = [ds.features.LABELS.EGOMOTION]
specs += [getattr(ds.features.CAMERA, x) for x in (
    'CAMERA_CROSS_LEFT_120FOV', 'CAMERA_FRONT_WIDE_120FOV',
    'CAMERA_CROSS_RIGHT_120FOV', 'CAMERA_FRONT_TELE_30FOV')]
assert len(set(specs)) == 5
assert set(specs).issubset(ds.feature_presence.columns)
assert ds.clip_index.index.is_unique
eligible = ds.feature_presence[specs].fillna(False).all(axis=1)
index = ds.clip_index.loc[eligible.reindex(ds.clip_index.index, fill_value=False)].copy()
index = index.loc[~index['chunk'].isin([0, 3119])]
collection = ds.data_collection.copy()
if collection.index.name != 'clip_id' and 'clip_id' in collection.columns:
    collection = collection.set_index('clip_id')
assert collection.index.is_unique
assert index.index.isin(collection.index).all(), 'metadata clip index mismatch: stop'
collection = collection.drop(columns=['clip_id'], errors='ignore')
table = index.join(collection, how='left', rsuffix='_collection')
table.index.name = 'clip_id'
table['t0_us'] = 5100000
table['status'] = 'candidate_only_pending_complete_usage_audit_and_selection'
table.to_csv(OUT/'precision-candidates.csv')
table.groupby('chunk').size().rename('eligible_clip_count').to_csv(OUT/'candidate-chunks.csv')
paths = ['features.csv', 'clip_index.parquet',
         'metadata/feature_presence.parquet', 'metadata/data_collection.parquet']
records = []
for info in api.get_paths_info('nvidia/PhysicalAI-Autonomous-Vehicles', paths,
                              repo_type='dataset', revision=REV):
    path = pathlib.Path(ds.download_file(info.path))
    content = path.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    assert len(content) == info.size
    if info.lfs:
        assert sha == info.lfs.sha256
    else:
        assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest() == info.blob_id
    records.append({'path':info.path, 'size':info.size, 'sha256':sha,
                    'git_blob':info.blob_id, 'cached_path':str(path)})
manifest = {'authority_source':'huggingface:nvidia/PhysicalAI-Autonomous-Vehicles',
            'authority_revision':REV, 'transport_source':'https://huggingface.co',
            'physical_ai_av_version':'0.2.0', 't0_us':5100000,
            'known_excluded_chunks':[0,3119],
            'complete_usage_audit':False, 'drive_disjoint_verified':False,
            'collection_columns':list(ds.data_collection.columns), 'metadata':records}
(OUT/'metadata-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({'metadata_total_bytes':sum(x['size'] for x in records),
                  'eligible_candidate_clips':len(table),
                  'eligible_candidate_chunks':int(table['chunk'].nunique()),
                  'out':str(OUT), 'status':'candidate_only'}, indent=2))
PY
```

本命令的候选输出不能直接成为测试名单。后续按精度决策锁定分层、样本量、角色、seed、失败/缺失处理及统计口径，完整排除 usage 后冻结名单。不能查看候选模型误差后更换样本。元数据只支持特征存在与粗粒度分层；`t0_us` 的四帧历史及 64 步未来完整性还需要下载后输入审计。

## 5. 用户命令 B：审定名单后先报价，再显式下载

前置产物必须由数据使用审计与精度决策给出，放在上一步目录中：

- `usage-ledger.json`：`dataset_revision`、`coverage_complete: true`、`used_clip_ids`（字符串列表）、`used_chunk_ids`（整数列表）、`audit_evidence`（非空证据路径列表）。`coverage_complete` 必须有实际审计支持，不得为了通过命令手填假值。
- `frozen-selection.json`：`dataset_revision`、`t0_us: 5100000`、`clip_ids`（非空且无重复）、`decision_id`、`usage_ledger_sha256`（审定账本的文件 SHA256）。角色和统计设计在决策中固定；命令只解析已冻结名单，不自行选 chunk。

账本证据和决策仍须人工/决策者复核。下面检查字段、摘要及重叠，不把一个布尔值当作独立性证明。第一次保持 `ALP15_EXECUTE_DOWNLOAD=0`，仅查询固定 revision 的 size/SHA256 并输出计划；明确决定执行已审定规模后，才把它改为 `1` 重跑。计划模式会读取已缓存元数据、查询文件信息，**不会下载 ZIP**。

```sh
env HF_ENDPOINT=https://huggingface.co ALP15_EXECUTE_DOWNLOAD=0 \
  /Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B - <<'PY'
import hashlib, importlib.metadata, json, os, pathlib, re, zipfile
from huggingface_hub import HfApi
import physical_ai_av as av

REV = '33f9bf447ed3bcb7d545ce13f4226f824214fafb'
OUT = pathlib.Path('/Users/USER/Downloads/alp15-precision-data-20261009')
assert importlib.metadata.version('physical-ai-av') == '0.2.0'
ledger_bytes = (OUT/'usage-ledger.json').read_bytes()
ledger = json.loads(ledger_bytes)
selection = json.loads((OUT/'frozen-selection.json').read_text())
assert ledger['dataset_revision'] == selection['dataset_revision'] == REV
assert ledger['coverage_complete'] is True and ledger['audit_evidence']
assert selection['usage_ledger_sha256'] == hashlib.sha256(ledger_bytes).hexdigest()
assert selection['decision_id'] and selection['t0_us'] == 5100000
clips = selection['clip_ids']
assert clips and all(isinstance(x,str) for x in clips) and len(set(clips)) == len(clips)
assert not set(clips).intersection(ledger['used_clip_ids'])
api = HfApi(endpoint='https://huggingface.co')
assert api.dataset_info('nvidia/PhysicalAI-Autonomous-Vehicles', revision=REV).sha == REV
ds = av.PhysicalAIAVDatasetInterface(revision=REV, cache_dir=str(OUT/'hf-cache'))
specs = [ds.features.LABELS.EGOMOTION]
specs += [getattr(ds.features.CAMERA, x) for x in (
    'CAMERA_CROSS_LEFT_120FOV', 'CAMERA_FRONT_WIDE_120FOV',
    'CAMERA_CROSS_RIGHT_120FOV', 'CAMERA_FRONT_TELE_30FOV')]
assert len(set(specs)) == 5
assert ds.feature_presence.loc[clips, specs].fillna(False).all().all()
chunks = sorted({int(ds.get_clip_chunk(x)) for x in clips})
assert not set(chunks).intersection({0,3119} | set(ledger['used_chunk_ids']))
paths = sorted({ds.features.get_chunk_feature_filename(c,f) for c in chunks for f in specs})
assert len(paths) == 5*len(chunks)
records = []
for info in api.get_paths_info('nvidia/PhysicalAI-Autonomous-Vehicles', paths,
                              repo_type='dataset', revision=REV):
    assert info.path in paths and info.lfs and re.fullmatch('[0-9a-f]{64}', info.lfs.sha256)
    records.append({'path':info.path, 'size':info.size, 'sha256':info.lfs.sha256})
assert {r['path'] for r in records} == set(paths)
plan = {'authority_source':'huggingface:nvidia/PhysicalAI-Autonomous-Vehicles',
        'authority_revision':REV, 'transport_source':'https://huggingface.co',
        'transport_revision':REV, 'physical_ai_av_version':'0.2.0',
        'selection_sha256':hashlib.sha256((OUT/'frozen-selection.json').read_bytes()).hexdigest(),
        'usage_ledger_sha256':hashlib.sha256(ledger_bytes).hexdigest(),
        'decision_id':selection['decision_id'], 't0_us':5100000,
        'chunks':chunks, 'clip_ids':clips, 'files':records,
        'total_zip_bytes':sum(r['size'] for r in records), 'drive_disjoint_verified':False}
print(json.dumps(plan, indent=2))
(OUT/'download-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
if os.environ.get('ALP15_EXECUTE_DOWNLOAD') != '1':
    print('PLAN ONLY: no ZIP downloaded')
    raise SystemExit(0)
verified = []
for record in records:
    path = pathlib.Path(ds.download_file(record['path']))
    assert path.stat().st_size == record['size']
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            sha.update(block)
    assert sha.hexdigest() == record['sha256']
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    verified.append({**record, 'cached_path':str(path), 'zip_crc_ok':True})
plan['verified_files'] = verified
plan['download_complete'] = True
plan['input_window_audit_complete'] = False
(OUT/'download-verified.json').write_text(json.dumps(plan, indent=2)+'\n')
print('Download bytes, SHA256 and ZIP CRC verified; input audit and dataset acceptance still pending.')
PY
```

逐个 `download_file` 复用 HF 缓存，并避开现有 Hub/Xet 与官方批量聚合进度条的已记录兼容问题，不新增 shim。中断后可重跑同一冻结名单和缓存，不删除缓存。任何 size/SHA256/ZIP CRC 失败都停止；不要改 revision、关闭校验或默默替换分块。`download-verified.json` 只在全部 ZIP 校验通过后写入，不代表解码、未来轨迹窗口、数据使用独立性、统计样本或模型精度已验收。传输到 L20 前还需保留缓存 `blobs/` 和 `snapshots/<revision>/` 的有效链接布局，不能只搬未解析的 symlink。

## 6. 本轮验证留痕与限制

- 环境：Mac，既有 Node 22.23.0；既有 Python 下载 venv；`physical-ai-av==0.2.0`。web-access 前置检查报告 Chrome 未连接，本任务使用公开静态 API 与只读官方认证，没有浏览器操作。
- 本地已完整读取两个相关下载/准备脚本及 `ALP15-SIM-DOWNLOAD-COMMANDS-20261008.md`；仿真来源核验不继承为原始 AV 数据镜像核验。
- 可复核查询：`curl -fsSL` 上表固定树/revision API；`curl -sSI` 固定镜像文件入口；官方 `HfApi().get_paths_info(repo_id, paths, repo_type='dataset', revision=REV)`。API 与 SHA256 输出保留在本次任务工具记录；正文没有保留账号、token 或个人信息。
- 本轮正文命令只做语法/静态检查，没有执行下载命令或生成审定 ledger/selection。未读取大 parquet，未下载容量示例 ZIP。
- 尚无可验收的完整国内传输源；尚未冻结新测试分块；尚未完成全历史 usage ledger、drive 层级独立性证明、输入窗口审计或独立精度验收。
