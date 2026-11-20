# 推理配置

配置页由 `tipdex config` 或模型服务的 `/config` 提供；下载 JSON 后通过 `--config` 加载。

| 参数 | 论文预设 `paper` | 鸡蛋预设 `egg` |
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

## 分数融合

模型包提供 `error_threshold_low=L`、`error_threshold_full=F`，且 `0<L<F`。令

```text
qbase = visual_offset + visual_scale*qv
x = clamp((e-L)/(F-L), 0, 1)
s = x*x*(3-2*x)
q = qbase - min(maximum_tactile_decrement*s, qbase/2)
```

无合格预测时 `e_for_gate=0`、`tactile_error=null`、`q=qbase`；不会把未核验情况显示为误差为零的成功预测。默认配置满足 `G(qv,0)=qv`，对 qv 单调不减、对误差单调不增。只有超过 L 的已核验误差才降低分数。

TACE 使用 `g=sigmoid(S-kappa*logit(q))`、`delta_U=g*b*tanh(R)`。R 与 S 共享包含当前状态、触觉特征和 Action Expert 特征的 trunk。q 在此路径停止梯度。每个快流步骤都恢复受保护位置，执行器在接受更新时再次核查可替换状态。
