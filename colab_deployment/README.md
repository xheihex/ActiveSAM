# ActiveSAM 的 Colab 部署

[单图笔记本](./ActiveSAM_single_image_colab.ipynb) 用于单图推理；[ADE20K 复现笔记本](./ActiveSAM_ADE20K_reproduction_colab.ipynb) 用于论文结果对照。两者共用 [adapter.py](./adapter.py)，直接加载作者的 ActiveSAM 类，不改写作者模型源码。

## 运行 ADE20K 复现

先把本目录部署文件推送到 [xheihex/ActiveSAM 的 colab-deployment 分支](https://github.com/xheihex/ActiveSAM/tree/colab-deployment)，然后在 [Colab 打开复现笔记本](https://colab.research.google.com/github/xheihex/ActiveSAM/blob/colab-deployment/colab_deployment/ActiveSAM_ADE20K_reproduction_colab.ipynb)。只需要这一个笔记本入口，其他代码自动从所选分支获取。

选择支持原生 BF16、显存足够的 GPU，在前两节表单填写参数并按顺序运行。默认执行前 20 张流程检查、完整 2,000 张默认配置，以及完整 2,000 张关闭 EPG 的对照。模型配置直接读取作者的 [base_config.py](../configs/base_config.py) 和 [cfg_ade20k.py](../configs/cfg_ade20k.py)，对照仅将 `tau` 从 0.1 改为 0。

**数据集 ZIP 路径在第 1 节的 `DATASET_PATH`，默认是 `/content/drive/MyDrive/datasets/ADE20K/ADE20K.zip`。** ZIP 内须包含 `images/validation` 与 `annotations/validation`，各 2,000 张；支持官方完整 ZIP 的目录前缀。程序只读取云盘 ZIP，在本地解压验证集。同一运行时重跑按源路径、大小和修改时间复用本地副本。

| 表单位置 | 参数 | 用途 |
| --- | --- | --- |
| 第 1 节 | `DATASET_PATH`、`WEIGHTS_PATH` | 云盘数据 ZIP 与 SAM 3 权重 |
| 第 1 节 | `OUTPUT_ROOT`、`RESUME_RUN_DIRECTORY` | 结果根目录；续跑填写上次打印的结果总目录，首次留空 |
| 第 1 节 | `WORDNET_DIR`、`WORK_DIR` | 已有 WordNet（留空则下载）、Colab 工作目录 |
| 第 2 节 | `GITHUB_URL`、`GIT_REF` | 你的仓库与 `colab-deployment` 分支，也可填固定提交 SHA |
| 第 2 节 | `RUN_NO_EPG`、`SMOKE_IMAGES` | 是否额外跑 EPG 对照、流程检查张数 |
| 第 2 节 | `SEED`、`CHECKPOINT_EVERY` | 随机种子、断点间隔（默认 20 张） |
| 第 2 节 | `INFERENCE_PREFETCH_WORKERS` | CPU 读图线程，默认 2；0 为串行，预取上限为线程数的两倍 |

代码默认隐藏，Colab 中点击“显示代码”可展开。VS Code 可执行相同代码，表单显示取决于编辑器支持。GPU 只由一个工作线程串行调用，CPU 预取与推理重叠；不并发调用作者的共享解码器。

## 指标与结果

沿用作者的 `class_maps()`、`decode()`、336/1008 分辨率处理、32 条提示分桶及现成 ADE20K 签名。原始标注 0 忽略，1..150 映射为 0..149，保留原图尺寸。轻量评测器累计整个验证集的 float32 交集、并集、预测和标注直方图，再按类别平均，不将逐图 mIoU 直接平均。

[论文 v2](https://arxiv.org/html/2606.16996v2) 表 10 的默认/关闭 EPG 参考 mIoU 为 **40.03 / 40.05**。结果仅检验 ADE20K 及 EPG 对照，不能代表整篇论文的所有实验。轻量评测器避免 Colab 编译 MMCV；它的 FPS 不能直接与作者 `Runner.test()` 的论文 FPS 比较。

每次结果总目录包含：

| 文件 | 内容 |
| --- | --- |
| `run.json` | 源码提交、输入路径、作者模型配置、PyTorch/CUDA 和 GPU |
| `comparison.csv/json` | 全量实验的 mIoU、论文差值和本机吞吐对照 |
| `smoke/`、`full/`、`no_epg/` 下的 `results.json` | 完成评测后的总体和逐类指标 |
| 各组的 `per_class.csv`、`per_image.csv` | 逐类指标、逐图尺寸/活跃类别数/耗时 |
| 各组的 `checkpoint.npz`、`settings.json` | 一个续跑断点，以及防止混入其他模型或输入的基本设置 |

`inference_fps` 包含模型前向、ECD、CUDA 同步和标签传回 CPU；`pipeline_fps` 还包含等待输入与指标计算，不重复计算与 GPU 重叠的预取时间。两者均不含断点写盘。续跑会记录 `resumed_from_images`；跨会话混合计时不适合严格比较速度。

## 中断与续跑

在 VS Code 点击“中断执行”或在 Colab 点击停止。程序保存已完成图片；当前未完成图片不计数，续跑时重算。当前 CUDA 调用需先收尾，期间会拒绝新推理或重建模型。等待收尾后重跑对应实验格即可继续，不必重新加载模型。

新会话从第 1 节运行，并将 `RESUME_RUN_DIRECTORY` 填为原结果总目录；保持数据、权重、源码提交和环境一致。保存断点时不要连续点击停止，强制重启可能损坏尚未写完的断点。本次精简后的断点格式与之前版本不同，应新建一次实验；原有结果无需删除。

## 本地维护

[reproduce.py](./reproduce.py) 只补充 ZIP 准备、轻量指标、输入预取、中断和断点、结果导出；模型推理使用作者实现。直接维护正式 [ADE20K 笔记本](./ActiveSAM_ADE20K_reproduction_colab.ipynb) 中的表单和入口，不保留执行输出。[test_reproduce.py](./test_reproduce.py) 使用内存图像与内存 I/O 检查指标、中断、续跑计数和笔记本结构，不创建验证目录。

从仓库根目录执行：

```bash
python -B -m unittest colab_deployment.test_reproduce -v
```

本地检查依赖 `numpy`、`Pillow`、`nbformat`，不能代替真实 SAM 3 GPU 运行。
