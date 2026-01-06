# main.py
import os
import re
import json
import time
import hashlib
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import OrderedDict

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional deps
try:
    import redis.asyncio as redis  # pip install redis
except Exception:
    redis = None

try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except Exception:
    BeautifulSoup = None

# =========================================================
# Config
# =========================================================

APP_NAME = "THE SHORT API"
PROMPT_VERSION_SHORT = "short.v1.0"
PROMPT_VERSION_DEEP = "deep.v1.0"
RETRIEVAL_VERSION = "r1"

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24h
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "2000"))

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "serper").strip().lower()  # serper|brave|none
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# ✅ 모델명은 환경변수로 교체 가능 (네가 쓰는 gemini-3.0-flash를 여기 넣으면 됨)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_MAX_TOKENS_SHORT = int(os.getenv("GEMINI_MAX_TOKENS_SHORT", "1600"))
GEMINI_MAX_TOKENS_DEEP = int(os.getenv("GEMINI_MAX_TOKENS_DEEP", "3200"))

REDIS_URL = os.getenv("REDIS_URL", "").strip()  # optional

FETCH_CONCURRENCY = int(os.getenv("FETCH_CONCURRENCY", "6"))
MAX_SOURCES = int(os.getenv("MAX_SOURCES", "12"))
MAX_EXCERPTS_PER_SOURCE = int(os.getenv("MAX_EXCERPTS_PER_SOURCE", "4"))
MAX_TOTAL_EXCERPTS = int(os.getenv("MAX_TOTAL_EXCERPTS", "48"))
MAX_FETCH_BYTES = int(os.getenv("MAX_FETCH_BYTES", str(800_000)))  # ~0.8MB
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "12"))

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (compatible; TheShortBot/1.0; +https://secondlook.onrender.com)"
)

# =========================================================
# Prompts (System + Templates)
# =========================================================

SYSTEM_PROMPT = """You are "THE SHORT" risk reviewer.

Tone:
•⁠  ⁠Cold, concise, decisive.
•⁠  ⁠Attack the logic, not the person.
•⁠  ⁠No profanity, insults, slurs, harassment.
•⁠  ⁠No memes, no roleplay, no emojis.

Safety:
•⁠  ⁠NOT investment advice. Do NOT tell the user to buy/sell/short/hold.
•⁠  ⁠Do NOT claim certainty. Use evidence and verification.

Truthfulness (critical):
•⁠  ⁠Use ONLY facts found in CONTEXT_EXCERPTS_JSON.
•⁠  ⁠Do NOT invent numbers, dates, events, or sources.
•⁠  ⁠If a needed number is missing, explicitly state it is missing and use next_metric.

Output:
•⁠  ⁠Return ONLY valid JSON (no markdown, no extra text).
•⁠  ⁠Follow the provided JSON schema exactly.
•⁠  ⁠Keep stable phrasing and ordering to improve caching consistency.
"""

SHORT_USER_PROMPT_TEMPLATE = """TASK:
Generate a "Short Report" in Korean as valid JSON.
Return:
•⁠  ⁠blindspot (single strongest missing assumption)
•⁠  ⁠questions (exactly 3)
•⁠  ⁠sources (0~6, only from CONTEXT_SOURCES_JSON)
•⁠  ⁠as_of, prompt_version, asset, ticker, company, conviction_original

Hard rules:
•⁠  ⁠Use ONLY facts from CONTEXT_EXCERPTS_JSON. No fabrication.
•⁠  ⁠If evidence is missing, say it is missing and ask for verification via next_metric.
•⁠  ⁠Questions must reference or quote the user's conviction_original.

INPUT:
prompt_version = "{prompt_version}"
asset = "{asset}"            # US|KR|COIN
ticker = "{ticker}"
company = "{company}"
conviction_original = "{conviction}"
now_iso = "{now_iso}"

CONTEXT_SOURCES_JSON:
{context_sources_json}

CONTEXT_EXCERPTS_JSON:
{context_excerpts_json}

OUTPUT JSON schema:
{{
  "prompt_version": "{prompt_version}",
  "as_of": "ISO8601 string",
  "asset": "US|KR|COIN",
  "ticker": "string",
  "company": "string",
  "conviction_original": "string",
  "blindspot": {{
    "title": "string",
    "value_line": "string",
    "detail": "string",
    "severity": "low|medium|high",
    "confidence": 0.0,
    "next_metric": "string",
    "sources": [
      {{ "title": "string", "url": "string" }}
    ]
  }},
  "questions": ["string", "string", "string"],
  "sources": [
    {{ "title": "string", "url": "string" }}
  ]
}}
Return ONLY JSON.
"""

DEEP_USER_PROMPT_TEMPLATE = """TASK:
Generate a "Deep Report" in Korean as valid JSON.
You must select 3 to 5 evidence slots among A-E.
If asset == "COIN", slot E is mandatory.

Slot definitions:
A: 수요/성장 (Revenue/Demand)
B: 마진/현금흐름 (Profitability/Cash Flow)
C: 경쟁/가격결정력 (Competition/Pricing Power)
D: 규제/법/집행/운영 리스크 (Regulatory/Legal/Execution)
E: 밸류/수급/공급/오버행 (Valuation/Supply/Overhang) — mandatory for COIN

Hard rules:
1) Every evidence must directly challenge the user's conviction sentence.
2) fact_line must be ONE line. If you cannot support it from CONTEXT_EXCERPTS_JSON, write "근거 수치/팩트가 확인되지 않았습니다." and set next_metric.
3) sources in each evidence must be selected ONLY from CONTEXT_SOURCES_JSON, and must match the evidence content.
4) Create exactly 3 counter_questions:
   - Each question must reference or quote a phrase from conviction_original.
   - Q1 forces a numeric threshold (반증 조건).
   - Q2 forces a time horizon / next check date (검증 시점).
   - Q3 forces an action plan under risk (대응 계획).

Ordering (stable):
•⁠  ⁠For US/KR: evidences order A -> B -> C -> D -> E (include only selected ones)
•⁠  ⁠For COIN: evidences order E -> A -> B -> D -> C (include only selected ones)
•⁠  ⁠counter_questions order: 조건 -> 시점 -> 대응

INPUT:
prompt_version = "{prompt_version}"
asset = "{asset}"            # US|KR|COIN
ticker = "{ticker}"
company = "{company}"
conviction_original = "{conviction}"
now_iso = "{now_iso}"

CONTEXT_SOURCES_JSON:
{context_sources_json}

CONTEXT_EXCERPTS_JSON:
{context_excerpts_json}

OUTPUT JSON schema:
{{
  "prompt_version": "{prompt_version}",
  "as_of": "ISO8601 string",
  "asset": "US|KR|COIN",
  "ticker": "string",
  "company": "string",
  "conviction_original": "string",
  "conviction_parsed": {{
    "claim": "string",
    "assumptions": ["string"],
    "time_horizon": "단기|중기|장기|불명"
  }},
  "evidences": [
    {{
      "slot_id": "A|B|C|D|E",
      "slot_name": "string",
      "title": "string",
      "fact_line": "string",
      "detail": "string",
      "severity": "low|medium|high",
      "confidence": 0.0,
      "next_metric": "string",
      "sources": [
        {{ "title": "string", "url": "string" }}
      ]
    }}
  ],
  "counter_questions": ["string", "string", "string"],
  "sources_top": [
    {{ "title": "string", "url": "string" }}
  ]
}}
Return ONLY JSON.
"""

# =========================================================
# FastAPI app
# =========================================================

app = FastAPI(title=APP_NAME)

if ALLOWED_ORIGINS.strip() == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Models
# =========================================================

class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    company: str = Field(default="Unknown", max_length=120)
    conviction: str = Field(..., min_length=1, max_length=400)
    locale: str = Field(default="ko-KR")
    asset: Optional[str] = Field(default=None, description="US|KR|COIN")
    force_refresh: bool = Field(default=False)

class DeepReportRequest(AnalyzeRequest):
    pass

# =========================================================
# Cache (Redis preferred, memory fallback)
# =========================================================

class TTLMemoryCache:
    """
    ✅ Fix for your Render crash:
    - This class ACCEPTS ttl_seconds/max_items so TTLMemoryCache(ttl_seconds=..., max_items=...) works.
    """
    def _init_(self, ttl_seconds: int = 86400, max_items: int = 2000):
        self.ttl_seconds = int(ttl_seconds)
        self.max_items = int(max_items)
        self._mem: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            item = self._mem.get(key)
            if not item:
                return None
            exp, val = item
            if time.time() > exp:
                self._mem.pop(key, None)
                return None
            # LRU touch
            self._mem.move_to_end(key, last=True)
            return val

    async def set(self, key: str, val: str, ttl: Optional[int] = None) -> None:
        ttl = int(ttl if ttl is not None else self.ttl_seconds)
        async with self._lock:
            exp = time.time() + ttl
            if key in self._mem:
                self._mem.pop(key, None)
            self._mem[key] = (exp, val)
            self._mem.move_to_end(key, last=True)

            # LRU trim
            while len(self._mem) > self.max_items:
                self._mem.popitem(last=False)

class Cache:
    def _init_(self):
        self.redis = None
        self.mem = TTLMemoryCache(ttl_seconds=CACHE_TTL_SECONDS, max_items=CACHE_MAX_ITEMS)

    async def init(self):
        if REDIS_URL and redis is not None:
            self.redis = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

    async def get(self, key: str) -> Optional[str]:
        if self.redis is not None:
            try:
                return await self.redis.get(key)
            except Exception:
                return await self.mem.get(key)
        return await self.mem.get(key)

    async def set(self, key: str, val: str, ttl: int) -> None:
        if self.redis is not None:
            try:
                await self.redis.setex(key, ttl, val)
                return
            except Exception:
                pass
        await self.mem.set(key, val, ttl)

cache = Cache()

@app.on_event("startup")
async def _startup():
    await cache.init()

# =========================================================
# Utilities
# =========================================================

def now_iso_kst() -> str:
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).isoformat()

def normalize_conviction(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t

def infer_asset(asset: Optional[str], ticker: str) -> str:
    if asset:
        a = asset.strip().upper()
        if a in ("US", "KR", "COIN"):
            return a

    t = ticker.strip()
    if re.fullmatch(r"\d{6}", t):
        return "KR"

    up = t.upper()
    if up in {"BTC", "ETH", "SOL", "XRP", "BNB", "WLD", "ADA", "DOGE", "AVAX", "LINK", "LTC"}:
        return "COIN"

    return "US"

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def cache_key(endpoint: str, asset: str, ticker: str, conviction_norm: str, prompt_version: str) -> str:
    raw = f"{endpoint}|{asset}|{ticker}|{conviction_norm}|{prompt_version}|{GEMINI_MODEL}|{RETRIEVAL_VERSION}"
    return "th:" + sha256_hex(raw)

def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)

def extract_domain(url: str) -> str:
    m = re.match(r"^https?://([^/]+)", url)
    return (m.group(1) if m else "").lower()

def is_probably_pdf(url: str, content_type: Optional[str]) -> bool:
    if url.lower().endswith(".pdf"):
        return True
    if content_type and "pdf" in content_type.lower():
        return True
    return False

def split_sentences(text: str) -> List[str]:
    t = text.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    chunks = re.split(r"\n{1,}", t)

    out: List[str] = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        parts = re.split(r"(?<=[\.\?\!])\s+|(?<=다\.)\s+|(?<=다\!)\s+|(?<=다\?)\s+", ch)
        for p in parts:
            s = p.strip()
            if 30 <= len(s) <= 320:
                out.append(s)

    seen = set()
    uniq: List[str] = []
    for s in out:
        key = s[:60]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq

def slot_to_tag(slot: str) -> str:
    return {"A": "demand", "B": "margin", "C": "competition", "D": "regulation", "E": "valuation"}.get(slot, "other")

def score_sentence(sentence: str, asset: str) -> Tuple[float, List[str], str]:
    s = sentence
    s_low = s.lower()

    score = 0.0
    tags: List[str] = []

    if re.search(r"\d", s):
        score += 4.0
    if re.search(r"(%|bps|\$|usd|krw|원|달러|억원|조원)", s_low):
        score += 2.0

    if re.search(r"(may|might|could|possible|추정|예상|가능성|전망)", s_low):
        score -= 1.0

    KW_A = ["매출", "성장", "수요", "출하", "주문", "사용자", "dau", "mau", "revenue", "growth", "deliveries", "shipments"]
    KW_B = ["마진", "영업이익", "순이익", "fcf", "현금흐름", "capex", "gross margin", "operating margin", "free cash flow", "cost"]
    KW_C = ["경쟁", "점유율", "가격", "단가", "asp", "pricing", "competition", "market share", "rival"]
    KW_D = ["규제", "제재", "조사", "소송", "벌금", "당국", "sec", "fss", "금감원", "lawsuit", "fine", "regulation"]
    KW_E_stock = ["밸류", "per", "p/e", "ev/ebitda", "희석", "증자", "오버행", "buyback", "dilution", "valuation", "multiple"]
    KW_E_coin = ["fdv", "유통", "총공급", "max supply", "unlock", "vesting", "인플레", "supply", "emission", "락업", "언락"]

    slot_scores: Dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}

    def bump(slot: str, kws: List[str], w: float):
        for kw in kws:
            if kw.lower() in s_low:
                slot_scores[slot] += w

    bump("A", KW_A, 1.0)
    bump("B", KW_B, 1.0)
    bump("C", KW_C, 1.0)
    bump("D", KW_D, 1.0)
    bump("E", KW_E_coin if asset == "COIN" else KW_E_stock, 1.2 if asset == "COIN" else 1.0)

    best_slot, best_val = max(slot_scores.items(), key=lambda x: x[1])
    score += best_val * 1.2

    sorted_slots = sorted(slot_scores.items(), key=lambda x: x[1], reverse=True)
    for slot, val in sorted_slots[:2]:
        if val > 0:
            tags.append(slot_to_tag(slot))

    confidence = "high" if re.search(r"\d", s) else "medium"
    return score, tags[:2], confidence

def tier_for_domain(asset: str, domain: str) -> str:
    d = domain.lower()
    primary_us = {"sec.gov"}
    primary_kr = {"dart.fss.or.kr", "kind.krx.co.kr", "krx.co.kr"}
    primary_coin = {"coinmarketcap.com", "coingecko.com"}

    reputable = {
        "reuters.com", "ft.com", "wsj.com", "bloomberg.com", "finance.yahoo.com",
        "investing.com", "marketwatch.com", "seekingalpha.com"
    }

    if asset == "US" and d in primary_us:
        return "primary"
    if asset == "KR" and d in primary_kr:
        return "primary"
    if asset == "COIN" and d in primary_coin:
        return "reputable"
    if d in reputable:
        return "reputable"
    return "other"

# =========================================================
# Search providers
# =========================================================

async def search_serper(query: str, num: int = 6) -> List[Dict[str, Any]]:
    if not SERPER_API_KEY:
        return []
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json", "User-Agent": USER_AGENT}
    payload = {"q": query, "num": num}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    out: List[Dict[str, Any]] = []
    for item in data.get("organic", []) or []:
        out.append({"title": item.get("title", ""), "url": item.get("link", ""), "snippet": item.get("snippet", "")})
    return [x for x in out if x.get("url")]

async def search_brave(query: str, count: int = 6) -> List[Dict[str, Any]]:
    if not BRAVE_API_KEY:
        return []
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json", "User-Agent": USER_AGENT}
    params = {"q": query, "count": str(count)}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()

    out: List[Dict[str, Any]] = []
    web = (data.get("web") or {}).get("results") or []
    for item in web:
        out.append({"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("description", "")})
    return [x for x in out if x.get("url")]

async def web_search(queries: List[str], per_query: int = 6) -> List[Dict[str, Any]]:
    if SEARCH_PROVIDER in ("none", "off"):
        return []
    results: List[Dict[str, Any]] = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        if SEARCH_PROVIDER == "brave":
            hits = await search_brave(q, count=per_query)
        else:
            hits = await search_serper(q, num=per_query)
        results.extend(hits)
    return results

def build_queries(asset: str, ticker: str, company: str) -> List[str]:
    base = f"{ticker} {company}".strip()
    if asset == "US":
        return [
            f"{base} 10-Q revenue operating margin free cash flow",
            f"{base} earnings release guidance margin",
            f"{base} competition pricing pressure market share",
            f"{base} lawsuit regulatory risk SEC",
            f"{base} valuation PE multiple",
        ]
    if asset == "KR":
        return [
            f"{ticker} {company} 분기보고서 매출 영업이익 DART",
            f"{company} 사업보고서 DART",
            f"{company} 공시 KIND KRX",
            f"{company} 경쟁사 점유율 가격",
            f"{company} 소송 규제 리스크",
        ]
    up = ticker.upper()
    return [
        f"{up} tokenomics FDV circulating supply",
        f"{up} unlock schedule vesting",
        f"{up} token distribution whitepaper",
        f"{up} regulatory investigation fine",
        f"{up} exchange delisting notice",
    ]

def dedup_and_rank_sources(asset: str, raw_hits: List[Dict[str, Any]], limit: int = MAX_SOURCES) -> List[Dict[str, Any]]:
    seen_url = set()
    by_domain_count: Dict[str, int] = {}

    def score_hit(h: Dict[str, Any]) -> float:
        url = h.get("url", "")
        dom = extract_domain(url)
        tier = tier_for_domain(asset, dom)
        s = 0.0
        s += 5.0 if tier == "primary" else 2.5 if tier == "reputable" else 0.5
        snippet = (h.get("snippet") or "").lower()
        if re.search(r"\d", snippet):
            s += 1.0
        return s

    ranked = sorted(raw_hits, key=score_hit, reverse=True)

    out: List[Dict[str, Any]] = []
    for h in ranked:
        url = (h.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        if url in seen_url:
            continue
        dom = extract_domain(url)
        if not dom:
            continue
        if by_domain_count.get(dom, 0) >= 2:
            continue

        seen_url.add(url)
        by_domain_count[dom] = by_domain_count.get(dom, 0) + 1

        out.append({
            "title": (h.get("title") or "").strip()[:180],
            "url": url,
            "publisher": dom,
            "published_at": "",
            "tier": tier_for_domain(asset, dom)
        })
        if len(out) >= limit:
            break
    return out

# =========================================================
# Fetch & Excerpt extraction
# =========================================================

async def fetch_text(url: str, sem: asyncio.Semaphore) -> Tuple[str, Optional[str]]:
    async with sem:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "/"},
            follow_redirects=True,
        ) as client:
            r = await client.get(url)
            ctype = r.headers.get("content-type")
            data = r.content[:MAX_FETCH_BYTES]

            if is_probably_pdf(url, ctype):
                return "", ctype

            if BeautifulSoup is None:
                return data.decode(errors="ignore"), ctype

            html = data.decode(errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside", "form"]):
                try:
                    tag.decompose()
                except Exception:
                    pass
            text = soup.get_text(separator="\n")
            lines = [ln.strip() for ln in text.splitlines()]
            lines = [ln for ln in lines if ln and len(ln) < 500]
            return "\n".join(lines), ctype

async def build_excerpts(asset: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(FETCH_CONCURRENCY)

    # add stable ids
    for i, s in enumerate(sources, start=1):
        s["id"] = f"s{i}"

    async def process_source(src: Dict[str, Any]) -> List[Dict[str, Any]]:
        url = src["url"]
        try:
            text, ctype = await fetch_text(url, sem)
        except Exception:
            return []

        if not text or is_probably_pdf(url, ctype):
            return []

        sents = split_sentences(text)
        scored: List[Tuple[float, str, List[str], str]] = []
        for sent in sents:
            sc, tags, conf = score_sentence(sent, asset)
            scored.append((sc, sent, tags, conf))
        scored.sort(key=lambda x: x[0], reverse=True)

        out: List[Dict[str, Any]] = []
        for sc, sent, tags, conf in scored[:MAX_EXCERPTS_PER_SOURCE]:
            out.append({
                "source_id": src["id"],
                "excerpt": sent,
                "tag": tags,
                "confidence": conf
            })
        return out

    chunks = await asyncio.gather(*(process_source(s) for s in sources), return_exceptions=True)
    excerpts: List[Dict[str, Any]] = []
    for ch in chunks:
        if isinstance(ch, Exception):
            continue
        excerpts.extend(ch)

    # stable-ish ordering: prefer digits first, then source_id
    def excerpt_rank(e: Dict[str, Any]) -> Tuple[int, str]:
        has_digit = 1 if re.search(r"\d", e.get("excerpt", "")) else 0
        return (-has_digit, e.get("source_id", ""))

    excerpts.sort(key=excerpt_rank)
    return excerpts[:MAX_TOTAL_EXCERPTS]

# =========================================================
# Gemini call (REST)
# =========================================================

class GeminiError(Exception):
    pass

def parse_json_strict(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    raise GeminiError("Model did not return valid JSON")

async def gemini_generate_json(user_prompt: str, max_tokens: int) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY is missing")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    params = {"key": GEMINI_API_KEY}

    payload: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": GEMINI_TEMPERATURE,
            "maxOutputTokens": max_tokens,
            "candidateCount": 1,
        },
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT * 2, headers={"User-Agent": USER_AGENT}) as client:
        r = await client.post(url, params=params, json=payload)
        if r.status_code >= 400:
            raise GeminiError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise GeminiError("Gemini response parse failed")

    return parse_json_strict(text)

# =========================================================
# Post-validate sources (no hallucinated URLs)
# =========================================================

def filter_sources_to_allowed(obj: Dict[str, Any], allowed_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    allowed_urls = {s["url"] for s in allowed_sources}
    url_to_title = {s["url"]: s["title"] for s in allowed_sources}

    def clean_source_list(lst: Any) -> List[Dict[str, str]]:
        if not isinstance(lst, list):
            return []
        out: List[Dict[str, str]] = []
        for x in lst:
            if not isinstance(x, dict):
                continue
            url = (x.get("url") or "").strip()
            if url in allowed_urls:
                out.append({"title": url_to_title.get(url, x.get("title", "") or ""), "url": url})
        # dedup
        seen = set()
        uniq = []
        for s in out:
            if s["url"] in seen:
                continue
            seen.add(s["url"])
            uniq.append(s)
        return uniq

    if "sources" in obj:
        obj["sources"] = clean_source_list(obj["sources"])
    if isinstance(obj.get("blindspot"), dict):
        obj["blindspot"]["sources"] = clean_source_list(obj["blindspot"].get("sources", []))

    if "sources_top" in obj:
        obj["sources_top"] = clean_source_list(obj["sources_top"])
    if isinstance(obj.get("evidences"), list):
        for ev in obj["evidences"]:
            if isinstance(ev, dict):
                ev["sources"] = clean_source_list(ev.get("sources", []))

    return obj

def ensure_three_questions(obj: Dict[str, Any], field: str, conviction: str) -> None:
    qs = obj.get(field)
    if not isinstance(qs, list):
        qs = []
    qs = [str(x) for x in qs if isinstance(x, str) and x.strip()]
    if len(qs) > 3:
        qs = qs[:3]
    while len(qs) < 3:
        if len(qs) == 0:
            qs.append(f"‘{conviction}’가 깨지는 숫자 기준(반증 조건)을 하나 정할 수 있습니까?")
        elif len(qs) == 1:
            qs.append(f"‘{conviction}’를 언제까지 검증할 겁니까? 다음 확인 시점을 날짜/이벤트로 고르세요.")
        else:
            qs.append(f"‘{conviction}’ 전제가 무너질 때 대응 계획은 무엇입니까? (손절/추가매수/유지 기준)")
    obj[field] = qs

# =========================================================
# Core pipeline
# =========================================================

async def retrieve_context(asset: str, ticker: str, company: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    queries = build_queries(asset, ticker, company)
    raw_hits: List[Dict[str, Any]] = []
    try:
        raw_hits = await web_search(queries, per_query=6)
    except Exception:
        raw_hits = []

    sources = dedup_and_rank_sources(asset, raw_hits, limit=MAX_SOURCES)
    if not sources:
        return [], []
    excerpts = await build_excerpts(asset, sources)
    return sources, excerpts

async def run_short(asset: str, ticker: str, company: str, conviction: str, force_refresh: bool) -> Dict[str, Any]:
    conviction_norm = normalize_conviction(conviction)
    ckey = cache_key("short", asset, ticker, conviction_norm, PROMPT_VERSION_SHORT)

    if not force_refresh:
        cached = await cache.get(ckey)
        if cached:
            return json.loads(cached)

    sources, excerpts = await retrieve_context(asset, ticker, company)

    context_sources_json = safe_json_dumps([
        {
            "id": s.get("id", ""),
            "title": s.get("title", ""),
            "url": s.get("url", ""),
            "publisher": s.get("publisher", ""),
            "published_at": s.get("published_at", ""),
            "tier": s.get("tier", "other"),
        }
        for s in sources
    ])
    context_excerpts_json = safe_json_dumps(excerpts)

    prompt = SHORT_USER_PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION_SHORT,
        asset=asset,
        ticker=ticker,
        company=company,
        conviction=conviction_norm,
        now_iso=now_iso_kst(),
        context_sources_json=context_sources_json,
        context_excerpts_json=context_excerpts_json
    )

    try:
        out = await gemini_generate_json(prompt, max_tokens=GEMINI_MAX_TOKENS_SHORT)
    except GeminiError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    out["prompt_version"] = PROMPT_VERSION_SHORT
    out["asset"] = asset
    out["ticker"] = ticker
    out["company"] = company
    out["conviction_original"] = conviction_norm
    out["as_of"] = out.get("as_of") or now_iso_kst()

    out = filter_sources_to_allowed(out, sources)
    ensure_three_questions(out, "questions", conviction_norm)

    if not isinstance(out.get("sources"), list) or len(out["sources"]) == 0:
        bs = out.get("blindspot") if isinstance(out.get("blindspot"), dict) else None
        if bs and isinstance(bs.get("sources"), list):
            out["sources"] = bs["sources"][:6]
        else:
            out["sources"] = []

    await cache.set(ckey, safe_json_dumps(out), CACHE_TTL_SECONDS)
    return out

async def run_deep(asset: str, ticker: str, company: str, conviction: str, force_refresh: bool) -> Dict[str, Any]:
    conviction_norm = normalize_conviction(conviction)
    ckey = cache_key("deep", asset, ticker, conviction_norm, PROMPT_VERSION_DEEP)

    if not force_refresh:
        cached = await cache.get(ckey)
        if cached:
            return json.loads(cached)

    sources, excerpts = await retrieve_context(asset, ticker, company)

    context_sources_json = safe_json_dumps([
        {
            "id": s.get("id", ""),
            "title": s.get("title", ""),
            "url": s.get("url", ""),
            "publisher": s.get("publisher", ""),
            "published_at": s.get("published_at", ""),
            "tier": s.get("tier", "other"),
        }
        for s in sources
    ])
    context_excerpts_json = safe_json_dumps(excerpts)

    prompt = DEEP_USER_PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION_DEEP,
        asset=asset,
        ticker=ticker,
        company=company,
        conviction=conviction_norm,
        now_iso=now_iso_kst(),
        context_sources_json=context_sources_json,
        context_excerpts_json=context_excerpts_json
    )

    try:
        out = await gemini_generate_json(prompt, max_tokens=GEMINI_MAX_TOKENS_DEEP)
    except GeminiError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    out["prompt_version"] = PROMPT_VERSION_DEEP
    out["asset"] = asset
    out["ticker"] = ticker
    out["company"] = company
    out["conviction_original"] = conviction_norm
    out["as_of"] = out.get("as_of") or now_iso_kst()

    out = filter_sources_to_allowed(out, sources)
    ensure_three_questions(out, "counter_questions", conviction_norm)

    if isinstance(out.get("evidences"), list):
        if len(out["evidences"]) > 5:
            out["evidences"] = out["evidences"][:5]
    else:
        out["evidences"] = []

    await cache.set(ckey, safe_json_dumps(out), CACHE_TTL_SECONDS)
    return out

# =========================================================
# Routes
# =========================================================

@app.get("/health")
async def health():
    return {
        "ok": True,
        "name": APP_NAME,
        "search_provider": SEARCH_PROVIDER,
        "has_serper_key": bool(SERPER_API_KEY),
        "has_brave_key": bool(BRAVE_API_KEY),
        "has_gemini_key": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL,
        "cache_ttl_seconds": CACHE_TTL_SECONDS
    }

@app.post("/v1/analyze")
async def analyze(req: AnalyzeRequest):
    ticker = req.ticker.strip()
    company = (req.company or "Unknown").strip()
    conviction_norm = normalize_conviction(req.conviction)

    if len(conviction_norm) < 10:
        raise HTTPException(status_code=400, detail="conviction must be at least 10 characters")

    asset = infer_asset(req.asset, ticker)
    return await run_short(asset, ticker, company, conviction_norm, req.force_refresh)

@app.post("/v1/short-report")
async def short_report(req: AnalyzeRequest):
    return await analyze(req)

@app.post("/v1/deep-report")
async def deep_report(req: DeepReportRequest):
    ticker = req.ticker.strip()
    company = (req.company or "Unknown").strip()
    conviction_norm = normalize_conviction(req.conviction)

    if len(conviction_norm) < 10:
        raise HTTPException(status_code=400, detail="conviction must be at least 10 characters")

    asset = infer_asset(req.asset, ticker)
    return await run_deep(asset, ticker, company, conviction_norm, req.force_refresh)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
