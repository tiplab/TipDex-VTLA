# 模型包与 Hugging Face 发布

已发布模型：[penguin3333333/TiP-Dex-EGG](https://huggingface.co/penguin3333333/TiP-Dex-EGG)。验收版本为 `57671fb7681fdab0f11dd97c683eb5d35f569307`，包含原始 9309、VASE、TACE、共享编码器触觉预测器，以及配套推理源码和安装包。安装与启动命令见 [README](../README.md)。

权重不进入源代码仓库。完整模型包包含以下内容：

```text
manifest.json
README.md
LICENSE
LICENSE_GEMMA.txt
backbone/config.json
backbone/params/                    # 原生 Orbax 参数与分片
backbone/assets/demo01_egg/norm_stats.json
backbone/assets/fastslow.json
tokenizer/paligemma_tokenizer.model
VASE/weights.npz
VASE/metadata.json
TACE/tace.msgpack
TACE/config.json
forecast/weights.pt
forecast/normalization.npz
forecast/contract.json
example/egg.npz
inference/TipDex-VTLA-0.2.0.zip
inference/tipdex_vtla-0.2.0-py3-none-any.whl
```

发布者使用 `tipdex-export` 打包已有权重，保留张量文件原始字节，并生成文件大小、SHA-256 及配套关系校验：

```bash
tipdex-export \
  --checkpoint /path/to/9309 \
  --vase /path/to/vase_artifact \
  --tace /path/to/tace_artifact \
  --forecast /path/to/shared_forecast_artifact \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --example /path/to/egg.npz \
  --output /path/to/model_bundle
```

导出器校验 TACE 绑定的 VASE、基座与归一化哈希。预测器加载时额外核查共享触觉编码器的参数摘要。打包只处理本地文件，不执行上传。

将完整模型包作为一个 Hugging Face **model** 仓库上传，保留全部 Orbax 分片和目录层级。记录上传后的 commit SHA，使用源码安装后的 `tipdex replay --model 仓库ID --revision 提交SHA --config paper` 完成从 Hub 下载到推理输出的发布验收。客户端拒绝用 `main` 等浮动分支代替固定提交。

`tipdex verify` 检查包内容；`tipdex replay` 验证真实权重的推理入口；`--stress-error` 额外构造大误差，用于检查最终分数下降、TACE 动作变化及受保护位置不变。该测试不构成触觉预测精度评测。
