# Qwen3-8B 微调操作指南（恒源云 3090 24G）

> 适用环境：恒源云 RTX 3090 24G · CUDA 12.8 · torch 2.9.0+cu128 · vLLM 0.9.0 · transformers<4.54
> 基座模型：`/hy-tmp/models/qwen3-8b`（16.4 GB，5 个 safetensors 分片）
> 更新：2026-11-07

---

## 0. 30 秒速览

| 项 | 值 |
|---|---|
| 推荐方案 | **QLoRA**（4bit 量化 + LoRA，显存 ~12G） |
| 训练框架 | LLaMA-Factory（省事）或 TRL SFTTrainer（可控） |
| 数据格式 | JSONL，`messages` 对话格式 |
| 输出 | LoRA adapter（~几十 MB），可被 vLLM 直接热加载 |
| 部署 | vLLM `--enable-lora`，模型名切换即生效 |

**一句话**：3090 上不要全参微调，QLoRA 训 8B 是唯一稳妥路径；数据决定成败，模板自动套，别手写拼接。

---

## 1. 方案选型

| 方案 | 显存 | 速度 | 结论 |
|---|---|---|---|
| 全参微调 | ~64G+ | 慢 | ❌ 3090 放不下 |
| LoRA（bf16 基座） | ~26-30G | 中 | ⚠️ 勉强，需关图形界面释放显存 |
| **QLoRA（4bit 基座）** | **~10-13G** | 中 | ✅ **首选**，效果≈LoRA 的 95% |

QLoRA 原理：基座模型 4bit 量化省显存，只训练注入的 LoRA 低秩矩阵（1-2% 参数），显存压力小、可插拔。

---

## 2. 环境准备（在恒源云实例上）

```bash
# ① 复用已有环境，装训练依赖（对齐 vLLM 的 transformers<4.54）
uv pip install peft trl datasets bitsandbytes -i https://mirrors.aliyun.com/pypi/simple/

# ② 验证（可选）
python -c "import peft,trl,datasets,bitsandbytes; print('ok')"
```

**版本约束（重要）**：
- `transformers` 保持 `<4.54`（已有），升级会撞 vLLM 的 aimv2 注册冲突
- `trl` 用 `0.11.x`（匹配 transformers 4.4x）；不要装最新 trl
- `peft` `>=0.12` 即可

**磁盘**：训练输出放 `/hy-tmp`（数据盘），别放系统盘 `/`（overlay 仅 30G）。

---

## 3. 数据准备与 Prompt 设计

### 3.1 数据格式（JSONL，一行一条）

```jsonl
{"messages": [{"role": "system", "content": "你是电商客服助手，回答简洁、友好，只回答商品相关。"}, {"role": "user", "content": "这个手机壳适配 iPhone 15 吗？"}, {"role": "assistant", "content": "适配的。这款手机壳专为 iPhone 15 设计，摄像头开孔精确对齐。"}]}
{"messages": [{"role": "user", "content": "明天会下雨吗？"}, {"role": "assistant", "content": "抱歉，我只能回答商品相关问题，天气请查询天气预报 App。"}]}
```

- `system` 可省略；`messages` 必须 `user` → `assistant` 成对
- 每条约 200-800 token 效果较好；过长（>2048）会拖慢训练
- **不要**在 JSON 里手动拼 `<|im_start|>` 等模板符（自动套）

### 3.2 Qwen3 对话模板与思考开关

训练时代码统一用 `tokenizer.apply_chat_template()`，自动套 Qwen3 模板：

```python
# 普通模式（不开思考，推荐先跑通）
tokenizer.apply_chat_template(ex["messages"], tokenize=False,
                              chat_template_kwargs={"enable_thinking": False})

# 思考模式（让模型先想后答）——assistant 内容需写成：
#   "<think>推理过程...</think>\n最终答案"
```

**坑**：Qwen3 模板默认 `enable_thinking=True`。如果训练数据里 assistant 没有 `<think>` 块，模板会强制把 `<think>\n` 作为生成前缀，导致训练 loss 偏高、输出异常。**不做思维链就显式关掉它。**

### 3.3 Prompt 工程要点

1. **system 写清角色 + 边界 + 语气**：比长指令有效，模型最吃人设约束
2. **给 2-3 个少样本示例**：同等效果下比堆描述好
3. **指令放最后**：user 消息 = 任务 + 上下文 + 问题，别把问题埋中间
4. **答案一致性**：同一类问题答案风格统一，否则模型学成墙头草
5. **覆盖拒绝场景**：加几条「不会/不能答」的样本，防止模型瞎编

### 3.4 数据清洗清单

- 去重（重复样本会让模型复读）
- 答案去头尾空白、统一标点
- 敏感/隐私内容剔除
- 正反例比例：正常回答 : 拒绝回答 ≈ 9 : 1
- 数据量：**500-5000 条**是这个量级模型的甜点；少于 200 条先人工核对，多于 2 万条考虑按领域抽样

---

## 4. 训练（两条路，二选一）

### 4.1 路径 A：LLaMA-Factory（推荐，少写代码）

```bash
cd /hy-tmp
git clone https://github.com/hiyouga/LLaMA-Factory && cd LLaMA-Factory
uv pip install -e ".[torch,bitsandbytes]" -i https://mirrors.aliyun.com/pypi/simple/

# 命令行训练（示例 yaml 在 examples/train_lora/qwen3_lora_sft.yaml）
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

`qwen3_lora_sft.yaml` 关键项（按需改）：

```yaml
model_name_or_path: /hy-tmp/models/qwen3-8b
dataset: your_dataset_name        # 数据放 data/ 目录并在 dataset_info.json 注册
template: qwen                    # 框架自带 qwen3 模板
finetuning_type: lora
quantization_bit: 4               # QLoRA
lora_rank: 16
lora_alpha: 32
output_dir: /hy-tmp/out/qwen3-8b-lora
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3.0
max_length: 2048
bf16: true
```

### 4.2 路径 B：手写 TRL SFT（完全可控）

保存为 `/hy-tmp/llm-deploy/sft.py`：

```python
import json
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

MODEL = "/hy-tmp/models/qwen3-8b"
OUT = "/hy-tmp/out/qwen3-8b-lora"
DATA = "/hy-tmp/data/train.jsonl"

# ---------- 4bit 量化基座 ----------
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto", torch_dtype="bfloat16")
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# ---------- 格式化为文本（关思考，见 3.2） ----------
def fmt(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False,
        chat_template_kwargs={"enable_thinking": False})
    return {"text": text}

train_data = load_dataset("json", data_files=DATA, split="train").map(fmt)

# ---------- 只对 assistant 回答部分算 loss ----------
resp_template = "<|im_start|>assistant"
collator = DataCollatorForCompletionOnlyLM(resp_template, tokenizer=tokenizer)

# ---------- LoRA ----------
lora = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()   # 期望 ~1-2% 可训练

# ---------- 训练 ----------
args = TrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # 等效 bs=8
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10, save_steps=200, save_total_limit=2,
    max_seq_length=2048,
    bf16=True, optim="paged_adamw_8bit",
    warmup_ratio=0.03, lr_scheduler_type="cosine",
)
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    args=args, train_dataset=train_data,
    data_collator=collator,
)
trainer.train()
trainer.save_model(OUT)
print("saved:", OUT)
```

运行（前台，别关终端）：

```bash
cd /hy-tmp/llm-deploy
python sft.py 2>&1 | tee /hy-tmp/logs/sft.log
```

---

## 5. 训练参数速查表

| 参数 | 建议值 | 说明 |
|---|---|---|
| `per_device_train_batch_size` | 1 | 3090 QLoRA 上限 |
| `gradient_accumulation_steps` | 8 | 等效 batch，越大越稳 |
| `learning_rate` | 1e-4 ~ 3e-4 | LoRA 用比全参大 10 倍 |
| `num_train_epochs` | 2 ~ 4 | 数据越少 epoch 越多 |
| `max_seq_length` | 2048 | 与你的业务长度匹配 |
| `lora_rank` | 16 | 数据少用 8，多用 32 |
| `lora_alpha` | = 2×r | 经验值 |
| `warmup_ratio` | 0.03 | 防初期震荡 |

**时长估算**（QLoRA，bs=1，3090）：1000 条约 1-2 小时；5000 条约 4-8 小时。可先跑 100 条试通。

---

## 6. 训练过程监控

```bash
# 看 loss 是否下降
grep -E "loss" /hy-tmp/logs/sft.log | tail -20

# 看显存
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

判断标准：
- **loss 从 1.x 缓慢降到 0.x** → 正常
- **loss 一开始就 <0.1** → 数据太简单或过拟合，加数据/减 epoch
- **loss 不降/震荡** → 学习率太大，降到 5e-5
- **显存报 OOM** → 关掉 JupyterLab 图形页面释放显存，或 `max_seq_length` 降到 1024

---

## 7. 评估（训练完必做）

```bash
# 用 t.py 换模型名快速聊一轮（注意：此时 vLLM 还没加载 LoRA）
python t.py qwen3-8b
```

正式评估清单：
- 10-20 条**没训过的**同类问题，看是否学会业务语言
- 3-5 条**越界问题**（不该答的），看是否会拒绝
- 1 条基座能力测试（数学/逻辑），确认没把通用能力训坏

---

## 8. 合并 LoRA（可选，用于部署/备份）

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "/hy-tmp/models/qwen3-8b", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "/hy-tmp/out/qwen3-8b-lora")
model = model.merge_and_unload()
model.save_pretrained("/hy-tmp/models/qwen3-8b-finetuned")
tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/models/qwen3-8b")
tokenizer.save_pretrained("/hy-tmp/models/qwen3-8b-finetuned")
```

合并后就是一个完整模型目录，可当新基座用。

---

## 9. 部署回 vLLM（LoRA 热加载，无需合并）

vLLM 0.9.0 原生支持 LoRA，**不合并也能用**：

```bash
pkill -f "vllm serve"
cd /hy-tmp/llm-deploy
PORT=8080 MODEL=/hy-tmp/models/qwen3-8b MAXLEN=4096 GPU_UTIL=0.90 \
EXTRA="--served-model-name qwen3-8b --api-key qwen10086 \
       --enable-lora --max-loras 4 \
       --lora-modules qwen3-8b-lora=/hy-tmp/out/qwen3-8b-lora" \
bash run.sh
```

调用方式（无需改 key/URL，只改 model 字段）：

```
model=qwen3-8b          # 基座
model=qwen3-8b-lora     # 微调版
```

> 若 QLoRA 训练时 4bit 基座 + LoRA，adapter 文件里不含基座权重，vLLM 直接挂基座 + adapter 即可。若合并过（第 8 节），直接 `MODEL=/hy-tmp/models/qwen3-8b-finetuned` 起服务，不用 lora 参数。

---

## 10. 常见问题排查

| 现象 | 原因 | 处理 |
|---|---|---|
| OOM | 显存不够 | 关 JupyterLab 图形页；降 `max_seq_length`；确认是 4bit 加载 |
| loss 为 NaN | 学习率过大 / bf16 溢出 | 降到 5e-5；换 `fp16: true` |
| 输出没变聪明 | LoRA 没生效 / 数据量太小 | 检查部署 `--lora-modules`；加到 1000+ 条 |
| 输出带 `<think>` | enable_thinking 未关 | 训练/推理都传 `enable_thinking: False` |
| 训练时 import vllm 冲突 | 环境混装 | 训练与 vLLM 分开环境，或用已有环境（transformers<4.54） |
| 中文乱码/重复 | 数据里有重复或编码问题 | 数据去重、统一 UTF-8 |

---

## 附录：数据样例（train.jsonl 片段）

```jsonl
{"messages": [{"role": "system", "content": "你是店铺客服，回答不超过 2 句话，语气友好。"}, {"role": "user", "content": "这个充电器支持快充吗？"}, {"role": "assistant", "content": "支持的，最大 65W 快充，20 分钟可充 50%。"}]}
{"messages": [{"role": "user", "content": "你们有实体店吗？"}, {"role": "assistant", "content": "目前只有线上店铺，支持 7 天无理由退换。"}]}
{"messages": [{"role": "user", "content": "帮我写个 Python 爬虫"}, {"role": "assistant", "content": "抱歉，我是店铺客服，只回答商品与订单相关问题。"}]}
```

**一句话总结**：数据做干净 → QLoRA 训 → vLLM `--enable-lora` 热加载 → 换 model 名即切换，全程不需要合并模型。
