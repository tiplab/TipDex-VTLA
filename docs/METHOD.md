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

论文流时间从噪声 τ=0 到动作 τ=1。TipDex-VTLA 基础策略内部使用 t=1−τ，十步积分的 dt=−0.1，六步后在 t≈0.4 缓存，快端完成后四步。

TACE 的 U 取每个快流步骤最后一个 Action Expert Transformer 输出，修正后经冻结动作投影层得到速度。残差和 gate 共享 64 维 GELU trunk。两个修正积分流使用 FP32，保留原始 BF16 参数值；修正差值加到原生流输出。已执行及不可替换位置在每一步恢复，防止注意力在动作位置之间传播修正而改变受保护命令。

<a id="configuration"></a>

## 配置与分数融合

配置页由 `tipdex config` 或模型服务的 `/config` 提供；下载 JSON 后通过 `--config` 加载。

| 参数 | 论文预设 `paper` | 部署预设 `tipdex` |
|---|---:|---:|
| 视觉分数 | `qv` | `0.6+0.4*qv` |
| `visual_offset` / `visual_scale` | 0 / 1 | 0.6 / 0.4 |
| `residual_bound` | 0.1 | 0.01 |
| `gate_kappa` | 1 | 1 |
| `maximum_tactile_decrement` | 0.1 | 0.1 |
| `forecast_horizon` | 30 | 5 |
| `tactile_fusion` | `context_concat` | `action_adapter` |
| 慢 / 快流步骤 | 6 / 4 | 6 / 4 |
| 动作块 | 30 帧，30 Hz | 30 帧，30 Hz |

`paper` 为默认配置，校验器要求其视觉映射、H、残差上界及上下文拼接符合该预设。页面手动修改参数后，配置标记为 `custom`。配置只在重新启动模型时加载，避免运行途中混用参数。

两个预设加载相同模型参数。循环预测解码器按 `forecast_horizon` 解码 H 帧；前五帧的解码路径不因 H 增大而改变。上下文拼接通过已加载的触觉投影参数，把共享触觉特征放在缓存视觉语言上下文之后、动作 token 之前；动作适配器预设使用动作侧交叉注意力。

### 分数融合

模型包提供 `error_threshold_low=L`、`error_threshold_full=F`，且 `0<L<F`。令

```text
qbase = visual_offset + visual_scale*qv
x = clamp((e-L)/(F-L), 0, 1)
s = x*x*(3-2*x)
q = qbase - min(maximum_tactile_decrement*s, qbase/2)
```

无合格预测时 `e_for_gate=0`、`tactile_error=null`、`q=qbase`；不会把未核验情况显示为误差为零的成功预测。默认配置满足 `G(qv,0)=qv`，对 qv 单调不减、对误差单调不增。只有超过 L 的已核验误差才降低分数。

TACE 使用 `g=sigmoid(S-kappa*logit(q))`、`delta_U=g*b*tanh(R)`。R 与 S 共享包含当前状态、触觉特征和 Action Expert 特征的 trunk。q 在此路径停止梯度。每个快流步骤都恢复受保护位置，执行器在接受更新时再次核查可替换状态。

## 触觉表示与转换

F1 输入为按 SDK 顺序排列的 **364 个 uint16 原始寄存器值**，有效范围为 `0…65535`。

```python
normalized = raw364.astype("float32") / 65535.0
heatmap = normalized.reshape(14, 26)  # 行优先；heatmap[r,c] = raw364[26*r+c]/65535
raw364_recovered = (heatmap.reshape(364) * 65535).round().astype("uint16")
```

预测器输出相同顺序的 364 个归一化值，展示时还原为 14×26 热图。触觉核验在同一表示下，对当前测量与对应预测的全部 364 个值计算 MAE。颜色仅用于显示，不参与误差计算。

**此转换不赋予寄存器八个解剖区域标签，也不要求对应五个指尖和三个掌部区域。** 热图位置表示寄存器序号，不表示物理触点位置；数值表示原始读数，不表示已标定的压力或力。接入有已知区域布局的传感器时，应在数据适配器中提供固定布局及归一化，并使用匹配的模型包。

策略编码使用最近 16 个 30 Hz 因果采样帧，按会话初始帧的中位数建立基线，再按 65535 缩放。Action Expert 与预测器复用同一个触觉编码器对象和参数。预测误差使用上面的 `raw/65535` 表示，预测值与测量值始终采用相同归一化。
