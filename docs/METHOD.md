# 论文推理定义与实现

此仓库提供模型推理、执行反馈协议和部署示例。

| 论文定义 | 实现 | 验证 |
|---|---|---|
| §3.1 视觉、语言、状态产生 c；VASE 分支位于触觉融合前 | `openpi/fastslow/model.py::_slow_snapshot` | 视觉前缀缓存；触觉上下文与动作 token 分离测试 |
| 触觉特征与 c 拼接，条件化 Action Expert | `openpi/models/pi0.py::embed_suffix` 的 `context_concat` | 拼接顺序、注意力方向、真实权重回放 |
| Action Expert / Forecaster 共享触觉编码器 | `forecast/shared_encoder.py` | 对象相同、特征数值相同、编码器参数不变 |
| Eq. (2) sigmoid(h(Pool(c))) | `VASE/head.py`、`VASE/runtime.py` | 加载实际 VASE 参数，缓存按 session/chunk 绑定 |
| 缓存中间流状态，继续剩余步骤 | `openpi/fastslow/model.py` | 六步后缓存，四步继续；原始时间及动作位置不变 |
| Eq. (3) 由 c 与当前状态、触觉、已知命令及时间预测 H 帧 | `forecast/model.py`、`runtime.py::issue_forecast` | H=30/5、共享编码器、完整命令时间表 |
| 选择最新合格的当前帧预测 | `forecast/verifier.py` | 拒绝事后签发、错位、未来、过期及异会话数据 |
| 干预动作必须与预测条件一致 | `forecast/execution.py` | 检查期间全部命令变化；拒绝缺失 ACK、驱动重启及改后恢复 |
| 相同归一化下 MAE；无合格预测时 e=0 | `forecast/verifier.py`、`TACE/score.py` | 当前帧即可核验，无须等待整个 H 帧；缺失证据显式回退 |
| G(qv,0)=qv，单调性 | `TACE/score.py` 的论文配置 | 恒等、单调及最大扣分测试 |
| Eq. (4) 共享 trunk、Gate Head、Residual Head | `TACE/refiner.py` | gate 公式、分量上界、q 停止梯度、实际动作变化 |
| 仅更新可替换、未执行位置 | `TACE/refiner.py`、`schedule.py` | 每一步恢复受保护状态；已提交位置及时间不变 |
| 每次接受动作更新后重新预测 | `runtime.py::fast_update` | 接受更新后触发；无可替换位置时不触发 |

## 实现约定

论文流时间从噪声 τ=0 到动作 τ=1。9309 内部使用 t=1−τ，十步积分的 dt=−0.1，六步后在 t≈0.4 缓存，快端完成后四步。

TACE 的 U 取每个快流步骤最后一个 Action Expert Transformer 输出，修正后经冻结动作投影层得到速度。残差和 gate 共享 64 维 GELU trunk。两个修正积分流使用 FP32，保留原始 BF16 参数值；修正差值加到原生流输出。已执行及不可替换位置在每一步恢复，防止注意力在动作位置之间传播修正而改变受保护命令。

固定 G 的解析式及预设见 [配置说明](CONFIGURATION.md)。触觉序列化和归一化见 [README](../README.md)。
