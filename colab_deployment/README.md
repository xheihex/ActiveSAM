# ActiveSAM 的 Colab 单图部署

本目录是 `xheihex/ActiveSAM` 分叉的部署代码。打开 `ActiveSAM_single_image_colab.ipynb`，选择 GPU 运行时，在表单填写 Drive 的权重、数据集和结果目录。默认的 Git 引用是这个分叉的 `colab-deployment` 分支。

`adapter.py` 直接载入作者的单图推理实现，在独立命名空间中用 PyTorch 的基类调用它。权重加载进度和 BF16 回退也由适配器处理。`requirements.txt` 列出单图依赖，笔记本通过 pip 安装预编译包。论文评测仍可使用仓库根目录的 `eval.py`。 [ActiveSAM 论文](https://arxiv.org/abs/2606.16996) · [作者仓库](https://github.com/VILA-Lab/ActiveSAM)
