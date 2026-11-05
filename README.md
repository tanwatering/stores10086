
## 当前部署（2026-11-05）
- 平台：恒源云 · RTX 4090 24G
- 服务端口：8080（不是 8000）
- 启动：PORT=8080 MODEL=/hy-tmp/models/qwen3-8b MAXLEN=4096 bash run.sh
- 公网接口：https://__________/v1/models
- 注意：根路径 `/` 返回 404 属正常，用 /v1/models 或 /docs
