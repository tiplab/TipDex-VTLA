# 部署与执行反馈接口

`tipdex serve` 加载模型并完成真实权重预热，提供本地 HTTP JSON 服务。模型预热、慢端与快端推理在同一个固定工作线程执行，触觉与执行确认通过独立接收线程持续送入。模型服务生成动作计划；机器人驱动负责限位、发送命令及报告 SDK 确认。所有接口采用同一个主机的单调时钟（Python `time.monotonic()`）；远端调用方需在采集/执行主机统一时间基准。

## 会话与数据

1. `POST /reset` 创建会话，返回 `session_id`。它清空模型缓存、预测、反馈和动作计划，不发送机器人复位命令。切换会话前，执行器须停止接收旧会话更新并完成命令队列交接。
2. `POST /feedback` 持续送入触觉帧、状态和实际 SDK ACK。开始推理前提供至少 16 个 30 Hz 触觉锚点建立会话基线，并提供覆盖最近 0.5 秒的状态及命令历史。
3. `POST /slow` 更新三路图像、状态及视觉前缀，得到 `chunk_id` 与 VASE 分数。
4. `POST /fast` 使用当前状态和触觉继续四个快流步骤，接受可替换位置的更新，然后重新签发触觉预测。
5. 驱动把动作放入不可替换队列前，调用 `POST /commit`。驱动只执行返回的 `replaced` 位置，并保持 `action_times`。

同一会话按顺序提交 slow/fast 请求。触觉和 ACK 在推理期间持续送入，不能等模型计算结束才采集。新 slow chunk 的交接由执行器负责，上一 chunk 已提交的命令继续受驱动队列保护；服务同时保护这些命令占用的时间区间，阻止新 chunk 覆盖。

### 触觉帧

```json
{
  "session_id": "当前会话 ID",
  "frames": [{
    "raw364": ["364 个 uint16 数值"],
    "valid": true,
    "acquisition_time": 100.0,
    "timestamp": 100.001
  }],
  "states": [{"timestamp": 100.001, "state": ["19 个数值"]}],
  "acks": []
}
```

上面数组中的文字表示数据长度，实际请求必须使用数值。`acquisition_time` 是采集时刻，`timestamp` 是接收时刻。当前状态的顺序为：机械臂 7 关节、手部 6 关节、末端 xyz 和 rpy。关节及姿态角单位为弧度，xyz 为米。

### 执行确认

机械臂与手部分别提供 ACK。它必须由实际 SDK 写入成功后的回调生成，不能从候选计划或测量位置倒推。

```json
{
  "kind": "sdk_write_ack", "schema": 1, "role": "arm",
  "host_boot_id": "主机启动 ID", "driver_boot_id": "驱动本次启动 ID",
  "units": "radians", "success": true, "sequence": 1,
  "positions": [0, 0, 0, 0, 0, 0, 0], "ack_at": 100.0, "lost": 0
}
```

`role=arm` 的位置为 7 维；`role=hand` 为 6 维，顺序为小指、无名指、中指、食指屈曲、拇指屈曲、拇指旋转。`host_boot_id` 取 `/proc/sys/kernel/random/boot_id`。sequence 严格连续递增；SDK 反馈应持续提供，即保持同一指令时也报告实际写入。相邻 ACK 间隔最多 50 ms。发现丢失、失败或驱动重启后，该会话的预测不再获得执行一致性确认，重新建立会话和完整历史后恢复核验。

核验同时检查历史命令与从预测锚点至当前触觉帧的全部命令变化。SDK ACK 表示命令被接受/写入，不表示机器人实际达到该位置。

### 慢更新

`POST /slow` 字段：`session_id`、`timestamp`、19 维 `state`、`images`。images 的键为 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`，每个值使用 `tipdex.wire.encode_rgb(rgb)` 产生的无损 PNG 对象：`{"encoding":"png","data":"base64内容"}`。也接受 HWC uint8 数值数组；实时调用推荐 PNG，以减少 JSON 数组解析开销。保持相机原始采集时间戳，不在编码后重新赋予观测时间。模型执行保持宽高比的 224×224 缩放及补边。

### 快更新与提交

`POST /fast` 字段：`session_id`、当前触觉对应的 `timestamp`、19 维当前 `state`、`executed_steps`，可选 30 元素布尔数组 `replaceable`。`executed_steps` 是执行器当前的下一可发送索引，包含已越过的过期位置；同一 chunk 中只增不减。返回 qv、触觉核验状态、MAE、q、动作及原执行时间、`replaced` 索引，以及更新后的 forecast。

`POST /commit` 字段：`session_id`、`chunk_id`、`indices`。这些位置此后不能被改写。TACE 每个流步骤的 mask 与接受计划时的 mask 都生效；推理期间新提交的位置也保留旧动作。

状态/触觉历史缺失、预测未赶上目标时刻或命令不一致时，响应明确给出 `forecast_unavailable_reason` / `tactile_reason`，使用 e=0 的定义回退。下一次动作更新会再次尝试预测。

## Python 接入

```python
from tipdex.policy import TipDexPolicy
from tipdex.runtime import ClosedLoop

policy = TipDexPolicy("/path/to/model_bundle", config="paper")
loop = ClosedLoop(policy)
session_id = loop.reset()
# 按上述协议持续调用 feed_frames、feed_state、acknowledge。
# 数据齐备后调用 slow_update、fast_update；提交驱动队列前调用 commit。
```

`Ctrl+C` 结束模型服务。关闭实际机器人执行器时应由设备驱动执行其停机流程。
