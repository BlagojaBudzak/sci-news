"""
Diagnostic: show the raw LLM output and locate JSON syntax errors.
Not part of the pipeline. Safe to delete after debugging.
"""
import json
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE

query = "sodium-ion batteries"
prompt = f"""
You are a scientific search assistant. A user wants to see a digest of
recent research on the following topic:

"{query}"

Please create a search plan for the OpenAlex scholarly database and a
deterministic prefilter. The plan must include:

1. search_phrases: a JSON array of 3-10 exact phrases that best capture the
   research topic. These phrases will be used verbatim as search queries for
   OpenAlex and arXiv. Use simple noun phrases or multi-word terms, not
   sentences.

2. positive_keywords: a JSON object mapping 3-10 important topical phrases
   to relevance weights (0.5 to 3.0). Higher weight means more important.

3. negative_keywords: a JSON object mapping 0-5 phrases that should be
   down-weighted because they are often false positives for this topic.
   Use weights from 0.5 to 3.0.

4. lookback_days: an integer from 1 to 60. Default 7 if unsure.

Output only a JSON object with those four keys. Do not include any other
text, explanations, or markdown.
"""

model_name = OLLAMA_MODEL
if model_name.startswith("ollama/"):
    model_name = model_name[len("ollama/"):]

url = f"{OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
payload = {
    "model": model_name,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": OLLAMA_TEMPERATURE,
    "stream": False,
}

print(f"Calling {url} with model={model_name} ...")
r = requests.post(url, json=payload, timeout=300)
r.raise_for_status()
content = r.json()["choices"][0]["message"]["content"]

print("=" * 70)
print("RAW OUTPUT:")
print("=" * 70)
print(content)
print("=" * 70)
print(f"Length: {len(content)} chars")
print("=" * 70)

try:
    json.loads(content)
    print("Direct json.loads: OK")
except json.JSONDecodeError as e:
    print(f"Direct json.loads: FAILED -> {e}")
    pos = e.pos
    lo = max(0, pos - 80)
    hi = pos + 80
    print(f"Error at position {pos}:")
    print(f"  ...{content[lo:pos]}<<<HERE>>>{content[pos:hi]}...")