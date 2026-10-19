"""Standalone embedding API probe; uses only Python's standard library.

Run from the project root:
    python example/others/test_embedding_api.py

Fill API_KEY below, or set LLM4AD_NODE_EMBEDDING_API_KEY / LLM4AD_API_KEY.
No LLM generation, evaluation, or experiment files are involved.
"""

import json
import math
import os
import time
import urllib.error
import urllib.request


# Edit these settings before running on the same server as your experiment.
API_KEY = "sk-fpYg2WDIAe8JKnlOQpP2BQROfzONm6HXFg14u8EAu5G2ymcM"
BASE_URL = "https://api.apilio.ai/v1"
MODEL = "text-embedding-v4"
DIMENSIONS = 64
TIMEOUT_SECONDS = 60
TEST_TEXT = "Select the next unvisited city using distance and lookahead."


def redact(text, api_key):
    return text.replace(api_key, "<API_KEY_REDACTED>") if api_key else text


def probe(api_key, dimensions=None):
    label = "默认维度（不发送 dimensions）" if dimensions is None else f"指定 {dimensions} 维"
    body = {"model": MODEL, "input": [TEST_TEXT], "encoding_format": "float"}
    if dimensions is not None:
        body["dimensions"] = dimensions
    request = urllib.request.Request(
        BASE_URL.rstrip("/") + "/embeddings",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    print(f"\n--- {label} ---", flush=True)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status = response.status
            raw = response.read().decode("utf-8", errors="replace")
        print(f"HTTP {status}，耗时 {time.perf_counter() - started:.2f} 秒")
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read(8192).decode("utf-8", errors="replace")
        finally:
            exc.close()
        print(f"HTTP {exc.code}，耗时 {time.perf_counter() - started:.2f} 秒")
        print("错误正文（最多 8192 字节）：")
        print(redact(raw, api_key) or "<空正文>")
        return {"ok": False, "status": exc.code, "dimensions": None}
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        print(f"连接失败，耗时 {time.perf_counter() - started:.2f} 秒")
        print(redact(f"{type(exc).__name__}: {exc}", api_key))
        return {"ok": False, "status": None, "dimensions": None}

    try:
        payload = json.loads(raw)
        items = payload["data"]
        if not isinstance(items, list) or len(items) != 1:
            raise ValueError("期望返回一个输入对应的一条 embedding")
        vector = items[0]["embedding"]
        if not isinstance(vector, list) or not vector:
            raise ValueError("embedding 不是非空数组")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) for value in vector):
            raise ValueError("embedding 包含非数值或非有限数值")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        print(f"响应格式异常：{exc}")
        print(redact(raw[:2000], api_key))
        return {"ok": False, "status": status, "dimensions": None}

    count = len(vector)
    print(f"返回模型：{payload.get('model', '<未提供>')}")
    print(f"返回向量维度：{count}")
    print(f"前 5 个分量：{vector[:5]}")
    if dimensions is not None and count != dimensions:
        print(f"失败：请求 {dimensions} 维，实际返回 {count} 维；服务可能忽略了 dimensions。")
        return {"ok": False, "status": status, "dimensions": count}
    print("成功")
    return {"ok": True, "status": status, "dimensions": count}


def main():
    api_key = (API_KEY.strip() or os.environ.get("LLM4AD_NODE_EMBEDDING_API_KEY", "").strip()
               or os.environ.get("LLM4AD_API_KEY", "").strip())
    if not api_key:
        print("请先填写脚本顶部的 API_KEY，或设置 LLM4AD_NODE_EMBEDDING_API_KEY。")
        return 2
    print(f"模型：{MODEL}；两个测试使用相同密钥和文本。")
    print("每种设置仅请求一次，不自动重试，以便观察原始故障。")
    default = probe(api_key)
    explicit = probe(api_key, DIMENSIONS)
    print("\n--- 结果判断 ---")
    if default["ok"] and explicit["ok"]:
        print(f"本次调用成功：模型可用，并且服务返回了指定的 {DIMENSIONS} 维向量。")
        print("短文本单请求成功，不代表长文本或高并发请求始终可用。")
    elif default["ok"]:
        print("默认维度成功，指定维度失败。可能是 dimensions 支持问题，也可能是临时服务故障；请重复测试确认。")
    elif explicit["ok"]:
        print("指定维度成功，默认维度失败。模型并非完全不可用；请结合错误正文检查默认设置或临时故障。")
    else:
        print("两个请求都失败；请依据 HTTP 状态及错误正文排查接口、密钥和模型通道。")
        print("401/403：检查鉴权或权限；404：检查地址或模型；429：检查限流或额度；")
        print("503：服务或上游暂不可用，不能单凭该状态认定模型不受支持。")
    return 0 if default["ok"] and explicit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
