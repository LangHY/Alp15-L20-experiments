# 用户指定NuRec魔搭仓库核查

日期：2026-10-11；只读小型公开API核查，无USDZ/MP4下载。

来源：https://www.modelscope.cn/datasets/nv-community/PhysicalAI-Autonomous-Vehicles-NuRec
仓库公开master根目录：sample_set、.gitattributes、clip_ratings_26.04.csv、LICENSE.pdf、README.md。
`sample_set/26.04_release`目录API TotalCount=1607，条目为场景UUID目录。
实查场景`00040136-e651-4abd-991d-0655ccda9430`含：

- UUID.usdz：1,955,916,532 bytes，约1.956 GB。
- camera_front_wide_120fov.mp4：38,142,684 bytes，约38.14 MB。

README使用公开API读取：
`https://modelscope.cn/api/v1/datasets/nv-community/PhysicalAI-Autonomous-Vehicles-NuRec/repo?Revision=master&FilePath=README.md`。
README说明：26.04版本1607场景、约20秒、重建来自6摄像头；完整仓库约2.9 TB（文档量级，非本轮逐文件求和）。
USDZ包含checkpoint、volume.nurec、map.xodr、rig_trajectories和clipgt/egomotion_estimate.parquet等，
四个前/侧相机frames目录提供首帧图像；场景目录通常提供一路前宽MP4。

结论：这是真实存在且有场景资产的国内NuRec镜像，但不是原始PhysicalAI相机分块镜像。
适合进一步研究NuRec/AlpaSim闭环渲染输入，不能直接替换已冻结四路真实视频与原egomotion精度输入。
不能因为有egomotion_estimate就假定其坐标、时间、GT定义与当前评测一致；需要单独验证契约。
只下载前宽MP4可用于单相机预览，不足以复现当前四相机输入。
当前没有修改949候选名单或切换下载来源；无NuRec下载/仿真运行授权扩展。
