import os, requests
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

def _openai_client(base_url: Optional[str] = None):
    if base_url:
        return OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY","EMPTY"))
    return OpenAI()

def call_chat(block: dict, system: str, user: str) -> str:
    prov = block["provider"].lower()
    if prov == "openai":
        client = _openai_client()
        r = client.chat.completions.create(
            model=block["model"],
            temperature=block.get("temperature",0.6),
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=block.get("max_tokens",512)
        )
        return r.choices[0].message.content

    if prov == "vllm":
        client = _openai_client(block.get("base_url"))
        r = client.chat.completions.create(
            model=block["model"],
            temperature=block.get("temperature",0.6),
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=block.get("max_tokens",512)
        )
        return r.choices[0].message.content

    if prov == "ollama":
        base = block.get("base_url","http://localhost:11434")
        url  = f"{base.rstrip('/')}/api/chat"
        payload = {
            "model": block["model"],
            "messages": [{"role":"system","content":system},{"role":"user","content":user}],
            "stream": False,
            "options": {"temperature": block.get("temperature",0.6),
                        "num_predict": block.get("max_tokens",512)}
        }
        @retry(stop=stop_after_attempt(4),
               wait=wait_exponential(min=1, max=20),
               retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)))
        def _post():
            r = requests.post(url, json=payload, timeout=180)
            r.raise_for_status()
            js = r.json()
            return js["message"]["content"] if "message" in js else js.get("response","")
        return _post()

    raise ValueError(f"Unknown provider: {prov}")
