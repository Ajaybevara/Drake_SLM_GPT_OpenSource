import asyncio
import base64
import io
import os
from functools import lru_cache
from urllib.parse import urlparse

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from PIL import Image

load_dotenv()
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# Never send local/LAN model traffic through a system proxy. This is important
# on managed Windows machines where HTTP(S)_PROXY may point at a corporate or
# sandbox proxy that cannot reach private addresses.
_vllm_host = urlparse(os.getenv("VLLM_API_BASE", "")).hostname
if _vllm_host:
    for _key in ("NO_PROXY", "no_proxy"):
        _hosts = [item.strip() for item in os.getenv(_key, "").split(",") if item.strip()]
        for _host in (_vllm_host, "127.0.0.1", "localhost"):
            if _host not in _hosts:
                _hosts.append(_host)
        os.environ[_key] = ",".join(_hosts)

from litellm import acompletion, completion


def _vllm_settings():
    api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
    api_key = os.getenv("VLLM_API_KEY", "empty")
    model = os.getenv("VLLM_MODEL", "qwen3-vl-8b")
    return api_base, api_key, model


def models_configured():
    return bool(os.getenv("VLLM_API_BASE") and os.getenv("VLLM_MODEL"))


async def llm_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
    api_base, api_key, model = _vllm_settings()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    response = await acompletion(
        model=f"openai/{model}",
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        timeout=float(os.getenv("VLLM_TIMEOUT_SECONDS", "300")),
        max_retries=2,
    )
    return response.choices[0].message.content


def _image_data_uri(image):
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def vision_complete(prompt, images, temperature=0.1):
    api_base, api_key, model = _vllm_settings()
    content = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": _image_data_uri(image)}}
        for image in images
    )
    response = completion(
        model=f"openai/{model}",
        messages=[{"role": "user", "content": content}],
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        timeout=float(os.getenv("VLLM_TIMEOUT_SECONDS", "300")),
        max_retries=2,
    )
    return response.choices[0].message.content


@lru_cache(maxsize=1)
def _embedding_model():
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    cache_dir = os.getenv("LOCAL_EMBEDDING_CACHE", "database/model_cache")
    return TextEmbedding(model_name=model_name, cache_dir=cache_dir)


async def local_embeddings(texts, **kwargs):
    def encode():
        return np.asarray(list(_embedding_model().embed(list(texts))), dtype=np.float32)

    return await asyncio.to_thread(encode)

