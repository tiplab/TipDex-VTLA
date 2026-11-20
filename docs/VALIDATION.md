# 推理验收

验收日期：2026-11-20。环境为独立 Python 3.11.15 虚拟环境，NumPy 1.26.4、JAX 0.5.3、PyTorch 2.7.1；GPU 为 NVIDIA A100 80GB。测试在源码目录之外运行安装后的 wheel，未启用 system site-packages。完整依赖版本见 `requirements-tested.txt`。

## 自动测试

**49 项通过**。覆盖分数恒等与单调性、gate 上界与停止梯度、当前帧核验、命令变化和缺失确认、共享编码器、30/5 帧预测、动作提交与跨 chunk 保护、模型包完整性、Hub 固定版本参数、配置 HTTP 接口、PNG 像素保持及固定推理工作线程。

## 真实权重回放

| 项目 | 论文配置 | 部署配置 |
|---|---:|---:|
| 策略参数 | 9309 | 9309 |
| 动作输出 | 30×13，有限值 | 30×13，有限值 |
| 触觉预测 | 30×364 | 5×364 |
| qv | 0.789391 | 0.789391 |
| 视觉基准 | 0.789391 | 0.915756 |
| 注入大误差后的 q | 0.689391 | 0.815756 |
| 误差引起的最大归一化动作变化 | 0.005059 | 0.001218 |
| 受保护位置变化 | 0 | 0 |
| 连续接受更新 / 重新预测次数 | 2 / 2 | 2 / 2 |

两种配置均确认共享同一个触觉编码器对象，特征最大差值为 4.77×10⁻⁷，编码器参数及原始权重文件保持不变；当前帧误差成功进入 TACE，已提交位置与原执行时间均保持不变。

## 模型服务

安装后的 `tipdex serve --config paper` 经同线程预热后，通过真实 HTTP slow/fast 请求输出动作和 30 帧预测。无损 PNG 保持 RGB 像素；该次慢端计算约 72 ms，观测到慢端响应约 179 ms。配置页、面板资源、预设校验与 VASE/TACE/q 状态数据均通过 HTTP 检查。该时延是上述测试环境的单次记录，不是跨硬件性能保证。

## 证据范围

回放使用真实模型参数和包内归档观测；执行确认由测试驱动显式构造，大误差测试使用 raw364 注入。测试不向机器人发送动作，不作为触觉精度、抓取成功率或论文实验指标。

本地模型包和 Hugging Face 发布版本均已验收。公开仓库 `penguin3333333/TipDex-VTLA` 的固定发布版本已完成匿名内容校验，版本号与逐文件证据见 `validation/huggingface/remote_integrity.json`：31 个清单文件全部通过 SHA-256 校验，安装包的 63 个模块及资源文件与已测试的安装版本逐字节一致。与发布清单匹配的真实权重在 paper / tipdex 两种配置下均通过两次闭环更新、重新预测、分数下降及已提交动作保护检查。命令见 [模型发布说明](MODEL_RELEASE.md)。

机器可读结果位于 `validation/installed_paper/report.json`、`validation/installed_tipdex/report.json`、`validation/http_smoke.json`、`validation/environment.json` 和 `validation/installed_tests.xml`。

Hugging Face 发布验收结果位于 `validation/huggingface/hub_validation_paper.json`、`validation/huggingface/hub_validation_tipdex.json` 和 `validation/huggingface/remote_integrity.json`。
