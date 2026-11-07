# 模型服务接入文档（给全栈开发）

> 更新时间：2026-11-07
> 对接人：全栈开发（Suerte）
> 部署人：tanα（模型部署）

---

## 1. 快速接入

| 项 | 值 |
|---|---|
| Base URL | `http://i-2.gpushare.com:20473/v1` |
| API Key | `qwen10086` |
| 模型名 | `qwen3-8b` |
| 服务状态 | `/health` 返回 200 即正常 |

**三行验证（curl）：**

```bash
# 健康检查（无需 key）
curl http://i-2.gpushare.com:20473/health

# 模型列表
curl http://i-2.gpushare.com:20473/v1/models -H "Authorization: Bearer qwen10086"

# 对话补全
curl http://i-2.gpushare.com:20473/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer qwen10086" \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"你好"}],"max_tokens":50}'
```

## 2. 认证

所有 `/v1/*` 请求头必须带：

```
Authorization: Bearer qwen10086
```

不带或 key 错误返回 `401 Unauthorized`。

## 3. Python 接入（openai SDK）

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://i-2.gpushare.com:20473/v1",
    api_key="qwen10086",
)

resp = client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100,
)
print(resp.choices[0].message.content)
```

## 4. 流式输出（推荐）

```python
stream = client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": "写一首诗"}],
    max_tokens=200,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
```

## 5. 重要提示

1. **是 HTTP 不是 HTTPS**：`https://` 页面的浏览器**不能直接 fetch** 这个地址（混合内容会被拦截）。**必须走你自己的后端代理转发**，后端服务器再以 HTTP 访问本服务。
2. **默认输出带思考块**：Qwen3 默认会输出 `<think>...</think>` 推理过程，`choices[0].message.reasoning_content` 字段单独存放。不需要思考时，在请求里加：

```json
"chat_template_kwargs": {"enable_thinking": false}
```

3. **上下文上限**：`max_model_len = 4096`（输入 + 输出 token 总和），超出报 400。
4. **鉴权已启用**：`qwen10086` 为当前 key，如泄露可在服务端一键更换。
5. **服务不持久**：按量付费实例，`/hy-tmp` 数据盘 24 小时不启动会清空；若服务挂了，联系部署人重启，接口契约不变。
6. **性能**：3090 实测基准数据待补充；纯文本 8B，单路延迟 TTFT 约几十毫秒级，请按真实压测数据做超时与并发设计。

## 6. 常见错误

| 现象 | 原因 | 处理 |
|---|---|---|
| 401 | key 缺失/错误 | 检查 `Authorization: Bearer qwen10086` |
| 404 | 路径不对 | 用 `/v1/models`、`/v1/chat/completions` |
| 400 | JSON 格式错 / 超长 | 校验请求体，控制 token 数 |
| timeout | 服务未启动或负载高 | 先查 `/health` |

## 7. 端点一览

| 端点 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查（无需 key） |
| `/v1/models` | GET | 模型列表 |
| `/v1/chat/completions` | POST | 对话补全（支持 stream） |
| `/v1/completions` | POST | 文本补全 |
| `/v1/embeddings` | POST | 向量嵌入 |
| `/docs` | GET | Swagger 接口文档（内网用） |
