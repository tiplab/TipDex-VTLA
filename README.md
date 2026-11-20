# TipDex_VTLA 官方推理示例

[项目主页与演示](https://tiplab.github.io/TipDex-VTLA/) · [源码](https://github.com/tiplab/TipDex-VTLA) · [TiP-Dex-EGG 模型](https://huggingface.co/penguin3333333/TiP-Dex-EGG)

本项目提供 TipDex_VTLA 的模型加载、慢快推理、VASE 视觉可靠性、触觉预测核验与 TACE 闭环修正。模型包直接包含 **9309 基座、VASE、TACE 和共享编码器触觉预测器**，通过本地目录或 Hugging Face 加载。

配套模型：[TiP-Dex-EGG](https://huggingface.co/penguin3333333/TiP-Dex-EGG)。以下命令固定到已经通过匿名下载和两种配置回放验收的模型版本。

## 安装与运行

环境：Linux、Python 3.11、NVIDIA CUDA 12 GPU。使用独立虚拟环境：

```bash
git clone https://github.com/tiplab/TipDex-VTLA.git
cd TipDex-VTLA
python3.11 -m venv .venv
source .venv/bin/activate
pip install ".[cuda]"

# 下载完整模型包；也可将 MODEL_DIR 设置为已有的本地模型目录。
export HF_MODEL_ID=penguin3333333/TiP-Dex-EGG
export HF_REVISION=57671fb7681fdab0f11dd97c683eb5d35f569307
export MODEL_DIR="$(python -c 'import os; from huggingface_hub import snapshot_download; print(snapshot_download(repo_id=os.environ["HF_MODEL_ID"], revision=os.environ["HF_REVISION"]))')"

tipdex verify --model "$MODEL_DIR"
CUDA_VISIBLE_DEVICES=0 tipdex replay --model "$MODEL_DIR" --config paper --output outputs/paper
```

`replay` 使用包内三路 RGB、19 维状态与 F1 原始触觉样例，运行真实模型权重，保存 `report.json` 和 `prediction.npz`。输出包括 `[30,13]` 动作、`[H,14,26]` 触觉预测、`q_v`、触觉误差和最终 `q`。回放中的 SDK 确认事件是明确构造的测试输入，不发送机器人命令；它验证推理链路，不作为预测精度或任务成功率评测。

Hugging Face 接口使用完整模型仓库及固定提交版本：

```bash
# HF_MODEL_ID、HF_REVISION 分别为模型仓库 ID 和 40 位 commit SHA
CUDA_VISIBLE_DEVICES=0 tipdex replay \
  --model "$HF_MODEL_ID" --revision "$HF_REVISION" --config paper
```

首次加载下载整个模型包；`--offline` 仅使用已缓存文件。每次加载检查文件 SHA-256、模型结构和模块配套关系。tokenizer 和归一化数据随模型包提供，不依赖训练服务器目录或隐式下载。

## 推理服务与配置页

```bash
# 论文配置：q=G(qv,e)，G(qv,0)=qv，H=30，b=0.1
CUDA_VISIBLE_DEVICES=0 tipdex serve --model "$MODEL_DIR" --config paper --port 8116

# 鸡蛋配置：视觉基准 0.6+0.4*qv，最多扣 0.1，H=5，b=0.01
CUDA_VISIBLE_DEVICES=0 tipdex serve --model "$MODEL_DIR" --config egg --port 8116
```

以上两条启动命令选择其一。服务在接收会话前加载权重并完成模型预热。

- `http://127.0.0.1:8116/dashboard`：展示 **VASE、TACE、q**、核验状态及预测热图。
- `http://127.0.0.1:8116/config`：选择配置、校验并下载 `inference.json`；启动时传入 `--config inference.json`。
- `tipdex config --port 8766`：单独开启配置页，无须加载模型。

启动独立示例服务后，可在同一主机用包内样例验证 HTTP 推理及面板数据：

```bash
python scripts/check_http.py --example "$MODEL_DIR/example/egg.npz" --url http://127.0.0.1:8116
```

该客户端使用模拟执行反馈，不向执行器发送动作。停止前台服务使用 `Ctrl+C`。接口、执行器接入与完整数据结构见 [部署说明](docs/DEPLOYMENT.md)，参数见 [配置说明](docs/CONFIGURATION.md)。

## F1 触觉表示与转换

F1 输入为按 SDK 顺序排列的 **364 个 uint16 原始寄存器值**，有效范围为 `0…65535`。

```python
normalized = raw364.astype("float32") / 65535.0
heatmap = normalized.reshape(14, 26)  # 行优先；heatmap[r,c] = raw364[26*r+c]/65535
raw364_recovered = (heatmap.reshape(364) * 65535).round().astype("uint16")
```

预测器输出相同顺序的 364 个归一化值，展示时还原为 14×26 热图。触觉核验在同一表示下，对当前测量与对应预测的全部 364 个值计算 MAE。颜色仅用于显示，不参与误差计算。

**此转换不赋予寄存器八个解剖区域标签，也不要求对应五个指尖和三个掌部区域。** 热图位置表示寄存器序号，不表示物理触点位置；数值表示原始读数，不表示已标定的压力或力。接入有已知区域布局的传感器时，应在数据适配器中提供固定布局及归一化，并使用匹配的模型包。

策略编码使用最近 16 个 30 Hz 因果采样帧，按会话初始帧的中位数建立基线，再按 65535 缩放。Action Expert 与预测器复用同一个触觉编码器对象和参数。预测误差使用上面的 `raw/65535` 表示，预测值与测量值始终采用相同归一化。

## 模块

```text
src/tipdex/
  VASE/       # 独立视觉可靠性头与慢端缓存
  TACE/       # 独立分数融合、共享 trunk、残差头、gate 及快流修正
  forecast/   # 共享编码器预测、时间对齐、执行确认与预测核验
  policy.py   # 9309 的慢快推理接口
  runtime.py  # 可替换动作计划、提交保护与更新后重新预测
  bundle.py   # 本地 / Hugging Face 模型加载与完整性校验
  static/     # 配置页和 VASE + TACE + q 前端
```

[方法对应表](docs/METHOD.md)列出论文推理定义、实现位置及验证方法。[模型发布说明](docs/MODEL_RELEASE.md)介绍模型包导出与 Hugging Face 仓库结构。已完成的检查与真实权重结果见 [推理验收](docs/VALIDATION.md)。

源代码遵循 [Apache-2.0](LICENSE)。上游代码与模型许可说明见 [第三方声明](NOTICE.md)。

## 项目主页

`docs/index.html`、`docs/content.js` 及 `docs/assets/` 提供项目主页、论文结果、架构图和真实/仿真演示。可在 GitHub 仓库中本地预览：

```bash
python -m http.server 8765 --directory docs
```

浏览器打开 `http://127.0.0.1:8765/`。
