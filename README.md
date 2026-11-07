# stores10086 · Qwen3-8B 推理服务

vLLM + Qwen3-8B，OpenAI 兼容 HTTP 接口。

## 当前部署（2026-11-07）

- 平台：恒源云 · **RTX 3090 24G**（CUDA 12.8 / 驱动 570.211.01）
- 引擎：vLLM 0.9.0 + torch 2.9.0+cu128
- 模型：`/hy-tmp/models/qwen3-8b`（16.4 GB，5 个 safetensors 分片）
- 服务端口：**8080**（不是 8000）
- 公网接口：`http://i-2.gpushare.com:20473/v1`（HTTP，不是 HTTPS）
  - `/v1/models` → 模型列表
  - `/v1/chat/completions` → 对话补全
  - `/health` → 健康检查
- 启动：

```bash
cd /hy-tmp/llm-deploy
PORT=8080 MODEL=/hy-tmp/models/qwen3-8b MAXLEN=4096 GPU_UTIL=0.90 \
EXTRA="--served-model-name qwen3-8b --api-key qwen10086" \
bash run.sh
```

- 验证：`python t.py qwen3-8b`（中文逐字输出 + TTFT）
- 注意：根路径 `/` 返回 404 属正常，用 `/v1/models` 或 `/docs`

## 环境安装（3090 + CUDA 12.8 专用）

**不要直接 `uv pip install vllm`**：PyPI 默认会装 torch 2.14.0+cu130 + vllm 0.33.0，在 CUDA 12.8 驱动下启动必挂（`NVIDIA driver on your system is too old`）。正确姿势：

```bash
# ① torch 固定 cu128 编译版
uv pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# ② vLLM 固定 0.9.0（阿里云镜像）
uv pip install vllm==0.9.0 -i https://mirrors.aliyun.com/pypi/simple/

# ③ transformers 必须 < 4.54（5.x 与 vLLM 0.9.0 的 aimv2 注册冲突）
uv pip install "transformers<4.54"

# ④ 验证
python -c "import vllm,torch;print(vllm.__version__,torch.__version__,torch.version.cuda,torch.cuda.is_available())"
# 期望：0.9.0 2.9.0+cu128 12.8 True
```

## 踩坑记录（3090/cu128 专场）

1. **PyPI 默认 torch 是 cu130**：vLLM 0.33.0 + torch 2.14.0+cu130 在 CUDA 12.8 驱动（570.211.01）下 `torch.cuda.is_available()` 返回 True，但引擎初始化报驱动过旧。必须用 cu128 编译的 torch（2.9.0）。
2. **vLLM 不能靠降级避坑**：0.7.3 不支持 Qwen3，只能选与 cu128 兼容的 0.9.0。
3. **uv 缓存无法复用**：若缓存里只有 cu130 的 torch/nvidia/triton wheel，装 cu128 版必须重新下载（缓存目录在 `archive-v0`）。
4. **transformers 5.x 撞 aimv2 注册**：vLLM 0.9.0 要求 `transformers>=4.51.1` 但无上限，5.20.2 导入即报 `ValueError: 'aimv2' already registered`，需 `transformers<4.54`。
5. **t.py 端口写死 8000**：恒源云服务端口是 8080，改 `base = "http://localhost:8080/v1"`。

## 未完成

- [ ] 关闭 Qwen3 thinking（输出带 `<think>` 块）或让调用方传 `enable_thinking: false`
- [ ] 量化对比（fp16 vs AWQ）
- [ ] 3090 并发压测（补 INTEGRATION.md 性能数据）
