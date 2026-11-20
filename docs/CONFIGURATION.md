# 推理配置入口

配置页由 `tipdex config` 或模型服务的 `/config` 提供。下载 `inference.json` 后，通过 `--config inference.json` 加载。

参数定义、任务配置和分数融合公式统一见[方法说明](METHOD.md#配置与分数融合)。配置在重新启动模型时加载。
