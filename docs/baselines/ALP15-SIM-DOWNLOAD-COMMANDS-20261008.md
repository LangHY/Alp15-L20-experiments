# 仿真资产下载指令（2026-10-08）

文件所有者：主智能体。仅提供用户执行命令；本轮没有启动任何场景或镜像大文件下载。

## 国内来源核验与版本决策

魔搭 [nv-community/PhysicalAI-Autonomous-Vehicles-NuRec](https://www.modelscope.cn/datasets/nv-community/PhysicalAI-Autonomous-Vehicles-NuRec) 的固定提交 `e4a281c0b5809d9425f06c1e5b10fd020003c4e2` 含26.04场景，不含旧26.01。选择同一clip的26.04新烟测，不能称为26.01复现。官方固定AlpaSim场景表支持26.04，配置边界见 `ALP15-ROOTLESS-SIM-20261008.md`。

已通过双方元数据核对：官方HF26.04提交 `5fe7a149b40ee83a2d3e09f7fe5dc2f217cabe11` 与魔搭提交中以下USDZ，大小1724125839字节、SHA256均为 `3e2eab16f8cd794f70b7f9b6b0dd290e66ea6f3141304c676a03515ebb0959ee`。这不是下载后的实际字节验收；下载后仍需执行校验。使用须遵守数据集许可，不代替用户接受许可或绕过访问限制。

## 1. Mac下载单个场景并校验

使用本机已安装的 ModelScope CLI，已实际执行 `modelscope download --help`（退出0）核实其新语法：位置参数repo_id、`--repo-type dataset`、`--local-dir`。旧版[官方CLI说明](https://github.com/modelscope/modelscope/blob/master/docs/source/command.md)的`--dataset/--local_dir`不适用于当前本机命令。未实际执行下载。不要删除缓存；失败可重跑同一固定来源命令。

```sh
(
set -e
SIM_SCENE_OUT='/Users/USER/Downloads/alp15-sim-scene-2604-20261008'
SIM_SCENE_FILE='sample_set/26.04_release/01d503d4-449b-46fc-8d78-9085e70d3554/01d503d4-449b-46fc-8d78-9085e70d3554.usdz'
modelscope download nv-community/PhysicalAI-Autonomous-Vehicles-NuRec \
  --repo-type dataset \
  --revision e4a281c0b5809d9425f06c1e5b10fd020003c4e2 \
  --local-dir "$SIM_SCENE_OUT" \
  "$SIM_SCENE_FILE"
test "$(stat -f '%z' "$SIM_SCENE_OUT/$SIM_SCENE_FILE")" = 1724125839
printf '3e2eab16f8cd794f70b7f9b6b0dd290e66ea6f3141304c676a03515ebb0959ee  %s\n' "$SIM_SCENE_OUT/$SIM_SCENE_FILE" | shasum -a 256 -c -
unzip -tq "$SIM_SCENE_OUT/$SIM_SCENE_FILE"
)
```

校验通过后先保留原文件，不在Mac展开并重新打包。内部metadata需在服务器读取并确认：scene_id为该clip、uuid为 `23dd34ea-a8d1-410c-aef7-d13f554cc4c9`、version_string为 `26.4.96-91b06fb8`。

## 2. NRE镜像：未找到已核实的国内镜像

本次公开搜索未找到可验证来源及digest一致性的国内 `nvidia/nre/nre-ga:26.04` 镜像。Docker Hub加速地址不等于NVCR镜像源，不提供猜测域名，也不把不同tag混用。下面是官方NVCR备用命令，**不是国内镜像指令**；由用户选择是否执行。

Mac只读获取manifest和config确认：

- manifest digest `sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284`，响应原始字节SHA256一致；linux/amd64。
- config digest `sha256:a6837f3877126947ff281ce9a6506261b65a2cb53d17f271da7852a9058c2289`。
- 压缩layer总大小14287662578字节，导出rootfs tar可能更大；需预留额外磁盘。
- Entrypoint=`/app/run`，WorkingDir=`/app/run.runfiles/_main`，Cmd=null。真正运行时必须复核初始化，不直接把展开目录当已验收运行环境。

以下借助 [crane export](https://github.com/google/go-containerregistry/blob/v0.20.3/cmd/crane/doc/crane_export.md) 导出合并rootfs，无需启动Docker daemon。需先自行准备可信的crane（例如Homebrew的 `brew install crane`）；不要把镜像逐层直接tar展开来代替合并whiteout。

```sh
(
set -e
SIM_IMAGE_OUT='/Users/USER/Downloads/alp15-nre-rootfs-20261008'
SIM_IMAGE_REF='nvcr.io/nvidia/nre/nre-ga@sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284'
mkdir "$SIM_IMAGE_OUT"
crane version > "$SIM_IMAGE_OUT/crane-version.txt"
crane manifest "$SIM_IMAGE_REF" > "$SIM_IMAGE_OUT/image-manifest.json"
crane config "$SIM_IMAGE_REF" > "$SIM_IMAGE_OUT/image-config.json"
crane export --platform linux/amd64 "$SIM_IMAGE_REF" "$SIM_IMAGE_OUT/nre-26.04-linux-amd64-rootfs.tar"
cd "$SIM_IMAGE_OUT"
shasum -a 256 nre-26.04-linux-amd64-rootfs.tar image-manifest.json image-config.json > SHA256SUMS.txt
shasum -a 256 -c SHA256SUMS.txt
tar -tf nre-26.04-linux-amd64-rootfs.tar > rootfs-members.txt
)
```

目录已存在时 `mkdir` 停止，避免覆盖既有导出；失败文件保留，由用户另选新目录再导出。crane下载依据digest验证镜像内容；本地产物SHA256用于传输验收，不冒称与镜像manifest digest相同。不要启用`--insecure`。tar成员安全、驱动绑定、独立rootfs CUDA、NRE真实render RPC与AlpaSim闭环仍是独立待验门禁。

## 当前执行边界

用户下载完成并告知路径后，再检查完整性、上传与服务器安全解包。当前不运行可能隐式下载大资产的AlpaSim wizard，即使 `run_method=NONE` 也不能直接运行。没有在服务器执行sudo或修改现有模型venv。
