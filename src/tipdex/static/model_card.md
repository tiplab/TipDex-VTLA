---
library_name: tipdex-vtla
pipeline_tag: robotics
tags:
- robotics
- vision-language-action
- tactile
- TipDex-VTLA
license: other
license_name: apache-2.0-and-gemma
license_link: LICENSE_GEMMA.txt
---
# TipDex-VTLA

用于 TipDex_VTLA 官方推理示例的完整模型包，包含 9309 原生 Orbax 基座参数、VASE、TACE、共享编码器触觉预测器、tokenizer、归一化参数与回放样例。

安装推理代码后：

```bash
tipdex verify --model /path/to/model_bundle
tipdex replay --model /path/to/model_bundle --config paper --output outputs/paper
tipdex serve --model /path/to/model_bundle --config paper --port 8116
```

输入为三路 RGB、19 维状态及最近 16 帧 F1 raw364。输出为 30×13 动作、H×14×26 预测热图及 VASE/TACE/q。论文配置 H=30；完整配置定义见[方法说明](https://github.com/tiplab/TipDex-VTLA/blob/main/docs/METHOD.md#configuration)。raw364 按寄存器顺序 reshape 为 14×26，数值除以 65535，不赋予物理触点或八个区域标签。

回放不发送机器人命令，使用明确构造的 SDK 确认事件验证推理和闭环协议。预测误差为同一归一化下当前帧 MAE，无合格预测时按 e=0 回退。

模型文件与组件关系由 manifest.json 校验。基座及派生参数的许可见 LICENSE 与 LICENSE_GEMMA.txt。源代码仓库提供完整接口和验证说明。
