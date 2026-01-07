# main.py
import os
import re
import json
import time
import hashlib
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional deps
try:
    import redis.asyncio as redis  # type: ignore
except Exception:
    redis = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None


# =========================================================
# Logging
# =========================================================
logger = logging.getLogger("the_short")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# =========================================================
# Env helpers
# =========================================================
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    try:
        return int(v.strip())
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    try:
        return float(v.strip())
    except Exception:
        return default


# =========================================================
# Config
# =========================================================
APP_NAME = "THE SHORT API"

PROMPT_VERSION_SHORT = env_str("PROMPT_VERSION_SHORT", "short.v1.1-evidence-first")
PROMPT_VERSION_DEEP = env_str("PROMPT_VERSION_DEEP", "deep.v1.1-evidence-first")
RETRIEVAL_VERSION = env_str("RETRIEVAL_VERSION", "r1")

CACHE_TTL_SECONDS = env_int("CACHE_TTL_SECONDS", 86400)

SEARCH_PROVIDER = env_str("SEARCH_PROVIDER", "serper").lower()
SERPER_API_KEY = env_str("SERPER_API_KEY", "")
BRAVE_API_KEY = env_str("BRAVE_API_KEY", "")

GEMINI_API_KEY = env_str("GEMINI_API_KEY", "")
GEMINI_MODEL = env_str("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_TEMPERATURE = env_float("GEMINI_TEMPERATURE", 0.2)
GEMINI_MAX_TOKENS_SHORT = env_int("GEMINI_MAX_TOKENS_SHORT", 1400)
GEMINI_MAX_TOKENS_DEEP = env_int("GEMINI_MAX_TOKENS_DEEP", 2600)

REDIS_URL = env_str("REDIS_URL", "")

FETCH_CONCURRENCY = env_int("FETCH_CONCURRENCY", 6)
MAX_SOURCES = env_int("MAX_SOURCES", 10)               # 비용/안정성 위해 조금 낮춤
MAX_EXCERPTS_PER_SOURCE = env_int("MAX_EXCERPTS_PER_SOURCE", 2)  # 비용/토큰 줄이기
MAX_TOTAL_EXCERPTS = env_int("MAX_TOTAL_EXCERPTS", 24)           # 비용/토큰 줄이기
MAX_FETCH_BYTES = env_int("MAX_FETCH_BYTES", 600_000)
HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 12.0)

# LLM 호출 최소 근거 개수(너무 적으면 LLM 호출 자체를 스킵하고 fallback)
MIN_EXCERPTS_FOR_LLM = env_int("MIN_EXCERPTS_FOR_LLM", 6)

ALLOWED_ORIGINS = env_str("ALLOWED_ORIGINS", "*")
USER_AGENT = env_str(
    "USER_AGENT",
    "Mozilla/5.0 (compatible; TheShortBot/1.0; +https://secondlook.onrender.com)"
)

KST = timezone(timedelta(hours=9))


# =========================================================
# Prompts (Evidence-first / Conviction is secondary)
# =========================================================
SYSTEM_PROMPT = """You are "THE SHORT" risk reviewer.

Tone:
- Cold, concise, decisive.
- Attack logic, not the person.
- No profanity, insults, slurs, harassment.
- No memes, no roleplay, no emojis.

Safety:
- NOT investment advice. Do NOT tell the user to buy/sell/short/hold.
- Do NOT claim certainty. Use evidence and verification language.

Truthfulness (critical):
- Use ONLY facts found in CONTEXT_EXCERPTS_JSON.
- Do NOT invent numbers, dates, events, or sources.
- If a needed fact/number is missing, explicitly say it is missing and use next_metric.

IMPORTANT (Evidence-first):
- Your primary job is to identify the asset's downside risks from context excerpts.
- The user's conviction sentence is NOT a data source. It is only used to phrase questions.
- Do NOT force every evidence to directly negate the conviction if the conviction is vague.

Output:
- Return ONLY valid JSON (no markdown, no extra text).
- Follow the provided JSON schema exactly.
"""

SHORT_USER_PROMPT_TEMPLATE = """TASK:
Generate a "Short Report" in Korean as valid JSON.

Goal (Evidence-first):
- Pick ONE most important downside risk ("blindspot") based on CONTEXT_EXCERPTS_JSON.
- Do NOT over-focus on the user's sentence. If the conviction is vague, still produce a solid blindspot from evidence.

Questions:
- Create exactly 3 questions.
- If possible, reference/quote the conviction_original.
- If conviction is vague, ask for measurable conditions (numbers), a next check date/event, and an action plan.

INPUT:
prompt_version = "{prompt_version}"
asset = "{asset}"
ticker = "{ticker}"
company = "{company}"
conviction_original = "{conviction}"
conviction_quality = "{conviction_quality}"  # low|medium|high
now_iso = "{now_iso}"

CONTEXT_SOURCES_JSON:
{context_sources_json}

CONTEXT_EXCERPTS_JSON:
{context_excerpts_json}

Return ONLY JSON.
"""

DEEP_USER_PROMPT_TEMPLATE = """TASK:
Generate a "Deep Report" in Korean as valid JSON.

Evidence slots:
A: 수요/성장 (Revenue/Demand)
B: 마진/현금흐름 (Profitability/Cash Flow)
C: 경쟁/가격결정력 (Competition/Pricing Power)
D: 규제/법/집행/운영 리스크 (Regulatory/Legal/Execution)
E: 밸류/수급/공급/오버행 (Valuation/Supply/Overhang) — mandatory for COIN

Evidence-first rules:
1) Produce 3~5 evidences about the asset's key downside risks from CONTEXT_EXCERPTS_JSON.
2) Do NOT force every evidence to directly negate the conviction sentence. Only link when it naturally fits.
3) If you cannot support from CONTEXT_EXCERPTS_JSON:
   - fact_line = "근거 수치/팩트가 확인되지 않았습니다."
   - detail explains what is missing
   - next_metric tells what to verify next
4) sources must be selected ONLY from CONTEXT_SOURCES_JSON.

Counter questions:
- Create exactly 3 counter_questions:
  Q1: numeric threshold (반증 조건)
  Q2: time horizon / next check date (검증 시점)
  Q3: action plan (대응 계획)
- Try to reference/quote conviction_original, but if conviction is vague, ask generic measurable questions.

Ordering:
- For US/KR: A -> B -> C -> D -> E (include 3~5)
- For COIN: E -> A -> B -> D -> C (E mandatory)

INPUT:
prompt_version = "{prompt_version}"
asset = "{asset}"
ticker = "{ticker}"
company = "{company}"
conviction_original = "{conviction}"
conviction_quality = "{conviction_quality}"  # low|medium|high
now_iso = "{now_iso}"

CONTEXT_SOURCES_JSON:
{context_sources_json}

CONTEXT_EXCERPTS_JSON:
{context_excerpts_json}

Return ONLY JSON.
"""


# =========================================================
# JSON Schemas (keep strict, but code has repair + fallback)
# =========================================================
SHORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt_version": {"type": "string"},
        "as_of": {"type": "string"},
        "asset": {"type": "string", "enum": ["US", "KR", "COIN"]},
        "ticker": {"type": "string"},
        "company": {"type": "string"},
        "conviction_original": {"type": "string"},
        "blindspot": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "value_line": {"type": "string"},
                "detail": {"type": "string"},
                "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                "confidence": {"type": "number"},
                "next_metric": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}, "url": {"type": "string"}},
                        "required": ["title", "url"],
                    },
                },
            },
            "required": ["title", "value_line", "detail", "severity", "confidence", "next_metric", "sources"],
        },
        "questions": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "url": {"type": "string"}},
                "required": ["title", "url"],
            },
        },
    },
    "required": ["prompt_version", "as_of", "asset", "ticker", "company", "conviction_original", "blindspot", "questions", "sources"],
}

DEEP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt_version": {"type": "string"},
        "as_of": {"type": "string"},
        "asset": {"type": "string", "enum": ["US", "KR", "COIN"]},
        "ticker": {"type": "string"},
        "company": {"type": "string"},
        "conviction_original": {"type": "string"},
        "conviction_parsed": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "time_horizon": {"type": "string", "enum": ["단기", "중기", "장기", "불명"]},
            },
            "required": ["claim", "assumptions", "time_horizon"],
        },
        "evidences": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "slot_id": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
                    "slot_name": {"type": "string"},
                    "title": {"type": "string"},
                    "fact_line": {"type": "string"},
                    "detail": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "confidence": {"type": "number"},
                    "next_metric": {"type": "string"},
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"title": {"type": "string"}, "url": {"type": "string"}},
                            "required": ["title", "url"],
                        },
                    },
                },
                "required": ["slot_id", "slot_name", "title", "fact_line", "detail", "severity", "confidence", "next_metric", "sources"],
            },
        },
        "counter_questions": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "sources_top": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "url": {"type": "string"}},
                "required": ["title", "url"],
            },
        },
    },
    "required": ["prompt_version", "as_of", "asset", "ticker", "company", "conviction_original", "conviction_parsed", "evidences", "counter_questions", "sources_top"],
}


# =========================================================
# Cache
# =========================================================
class TTLCache:
    def __init__(self) -> None:
        self._mem: Dict[str, Tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            v = self._mem.get(key)
            if not v:
                return None
            exp, val = v
            if time.time() > exp:
                self._mem.pop(key, None)
                return None
            return val

    async def set(self, key: str, val: str, ttl: int) -> None:
        async with self._lock:
            self._mem[key] = (time.time() + ttl, val)

class Cache:
    def __init__(self) -> None:
        self.redis = None
        self.mem = TTLCache()

    async def init(self) -> None:
        if REDIS_URL and redis is not None:
            try:
                self.redis = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
            except Exception:
                self.redis = None

    async def close(self) -> None:
        if self.redis is not None:
            try:
                await self.redis.close()
            except Exception:
                pass

    async def get(self, key: str) -> Optional[str]:
        if self.redis is not None:
            try:
                v = await self.redis.get(key)
                if v is not None:
                    return v
            except Exception:
                pass
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


# =========================================================
# FastAPI app
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache.init()
    yield
    await cache.close()

app = FastAPI(title=APP_NAME, lifespan=lifespan)

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
# Utils
# =========================================================
def now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")

def normalize_conviction(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def infer_asset(asset: Optional[str], ticker: str) -> str:
    if asset:
        a = asset.strip().upper()
        if a in ("US", "KR", "COIN"):
            return a

    t = ticker.strip()
    if re.fullmatch(r"\d{6}", t):
        return "KR"

    up = t.upper()
    if up in {"BTC", "ETH", "SOL", "XRP", "BNB", "WLD", "ADA", "DOGE", "AVAX", "LINK"}:
        return "COIN"
    return "US"

def conviction_quality(conviction: str) -> str:
    """
    low/medium/high: 문장에 숫자/조건/검증가능성이 얼마나 있는지 대충 점수화.
    (A 안정화에서는 품질이 낮아도 팩트 리포트는 만들게 하고,
     질문만 더 템플릿/측정형으로 유도)
    """
    c = conviction.strip()
    score = 0
    if len(c) >= 35:
        score += 1
    if re.search(r"\d|%|\$|원|달러|bps|분기|Q[1-4]|FY|년|월|주", c, re.IGNORECASE):
        score += 2
    if re.search(r"(이면|하면|까지|전까지|before|after|if|when|조건|반증|깨지|검증)", c, re.IGNORECASE):
        score += 1

    if score >= 3:
        return "high"
    if score >= 2:
        return "medium"
    return "low"

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


# =========================================================
# Sentence extraction heuristics
# =========================================================
def split_sentences(text: str) -> List[str]:
    t = re.sub(r"[ \t]+", " ", text.replace("\r", "\n"))
    chunks = re.split(r"\n{1,}", t)
    out: List[str] = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        parts = re.split(r"(?<=[\.\?\!])\s+|(?<=다\.)\s+|(?<=다\!)\s+|(?<=다\?)\s+", ch)
        for p in parts:
            s = p.strip()
            if 30 <= len(s) <= 280:
                out.append(s)
    seen = set()
    uniq: List[str] = []
    for s in out:
        k = s[:60]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)
    return uniq

def score_sentence(sentence: str, asset: str) -> float:
    s_low = sentence.lower()
    score = 0.0
    if re.search(r"\d", sentence):
        score += 4.0
    if re.search(r"(%|bps|\$|usd|krw|원|달러|억원|조원)", s_low):
        score += 2.0
    if re.search(r"(may|might|could|possible|추정|예상|가능성|전망)", s_low):
        score -= 1.0

    # Slot-ish keywords (very rough)
    KW = [
        "revenue","sales","growth","deliveries","shipments","demand","order",
        "margin","profit","operating","free cash flow","fcf","capex",
        "competition","pricing","market share","asp",
        "lawsuit","sec","regulation","fine","investigation",
        "valuation","multiple","p/e","ev/ebitda","dilution","buyback",
        "tokenomics","fdv","circulating","unlock","vesting","supply",
        "매출","성장","수요","출하","주문","마진","영업이익","현금흐름",
        "경쟁","점유율","가격","규제","소송","벌금","밸류","per","언락","유통","총공급"
    ]
    for kw in KW:
        if kw in s_low:
            score += 0.5
    return score


# =========================================================
# Search providers
# =========================================================
async def search_serper(query: str, num: int = 5) -> List[Dict[str, Any]]:
    if not SERPER_API_KEY:
        return []
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    payload = {"q": query, "num": num}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    out: List[Dict[str, Any]] = []
    for item in data.get("organic", []) or []:
        link = (item.get("link") or "").strip()
        if link.startswith("http"):
            out.append({
                "title": (item.get("title") or "").strip(),
                "url": link,
                "snippet": (item.get("snippet") or "").strip()
            })
    return out

async def search_brave(query: str, count: int = 5) -> List[Dict[str, Any]]:
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
        link = (item.get("url") or "").strip()
        if link.startswith("http"):
            out.append({
                "title": (item.get("title") or "").strip(),
                "url": link,
                "snippet": (item.get("description") or "").strip()
            })
    return out

async def web_search(queries: List[str], per_query: int = 5) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        try:
            if SEARCH_PROVIDER == "brave":
                hits = await search_brave(q, count=per_query)
            else:
                hits = await search_serper(q, num=per_query)
        except Exception:
            hits = []
        results.extend(hits)
    return results

def build_queries(asset: str, ticker: str, company: str) -> List[str]:
    """
    ✅ 핵심: conviction(사용자 문장)과 무관하게 '기본 팩트 체크'를 항상 돈다.
    (문장 과몰입 원인을 서버/검색 단계에서부터 차단)
    """
    base = f"{ticker} {company}".strip()
    if asset == "US":
        return [
            f"{base} latest earnings revenue margin free cash flow",
            f"{base} guidance demand deliveries shipments",
            f"{base} regulatory lawsuit investigation SEC",
        ]
    if asset == "KR":
        return [
            f"{ticker} {company} 분기보고서 매출 영업이익 DART",
            f"{company} 공시 KIND KRX",
            f"{company} 소송 규제 리스크",
        ]
    up = ticker.upper()
    return [
        f"{up} tokenomics circulating supply FDV",
        f"{up} unlock schedule vesting",
        f"{up} regulatory risk investigation",
    ]

def tier_for_domain(asset: str, domain: str) -> str:
    d = domain.lower()
    primary_us = {"sec.gov"}
    primary_kr = {"dart.fss.or.kr", "kind.krx.co.kr", "krx.co.kr"}
    primary_coin = {"coinmarketcap.com", "coingecko.com"}
    reputable = {
        "reuters.com","ft.com","wsj.com","bloomberg.com","finance.yahoo.com",
        "investing.com","marketwatch.com"
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

def dedup_and_rank_sources(asset: str, raw_hits: List[Dict[str, Any]], limit: int = MAX_SOURCES) -> List[Dict[str, Any]]:
    by_domain_count: Dict[str, int] = {}
    seen_url = set()
    out: List[Dict[str, Any]] = []

    def score_hit(h: Dict[str, Any]) -> float:
        url = h.get("url", "")
        dom = extract_domain(url)
        tier = tier_for_domain(asset, dom)
        s = 5.0 if tier == "primary" else 2.5 if tier == "reputable" else 0.5
        if re.search(r"\d", (h.get("snippet") or "")):
            s += 1.0
        return s

    ranked = sorted(raw_hits, key=score_hit, reverse=True)

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
            "id": f"s{len(out)+1}",
            "title": (h.get("title") or "").strip()[:180],
            "url": url,
            "publisher": dom,
            "published_at": "",
            "tier": tier_for_domain(asset, dom),
        })
        if len(out) >= limit:
            break
    return out


# =========================================================
# Fetch & excerpts
# =========================================================
async def fetch_text(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> Tuple[str, Optional[str]]:
    async with sem:
        r = await client.get(url, headers={"Accept": "*/*"})
        ctype = r.headers.get("content-type")

        if r.status_code >= 400:
            return "", ctype

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
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True
    ) as client:

        async def process_source(src: Dict[str, Any]) -> List[Dict[str, Any]]:
            try:
                text, ctype = await fetch_text(client, sem, src["url"])
            except Exception:
                return []
            if not text or is_probably_pdf(src["url"], ctype):
                return []

            sents = split_sentences(text)
            scored: List[Tuple[float, str]] = []
            for sent in sents:
                sc = score_sentence(sent, asset)
                scored.append((sc, sent))
            scored.sort(key=lambda x: x[0], reverse=True)

            out_local: List[Dict[str, Any]] = []
            for sc, sent in scored[:MAX_EXCERPTS_PER_SOURCE]:
                out_local.append({
                    "source_id": src["id"],
                    "excerpt": sent,
                })
            return out_local

        chunks = await asyncio.gather(*[process_source(s) for s in sources], return_exceptions=True)

    out: List[Dict[str, Any]] = []
    for ch in chunks:
        if isinstance(ch, Exception):
            continue
        out.extend(ch)

    # 숫자 포함 문장 우선
    out.sort(key=lambda e: (-(1 if re.search(r"\d", e.get("excerpt", "")) else 0), e.get("source_id", "")))
    return out[:MAX_TOTAL_EXCERPTS]


# =========================================================
# Gemini (REST) + JSON repair
# =========================================================
class GeminiError(Exception):
    pass

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_first_json_block(s: str) -> Optional[str]:
    s = _strip_code_fences(s)
    starts = [i for i in [s.find("{"), s.find("[")] if i != -1]
    if not starts:
        return None
    start = min(starts)

    stack: List[str] = []
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
            continue
        if ch in "}]":
            if not stack:
                continue
            opener = stack[-1]
            if (opener == "{" and ch == "}") or (opener == "[" and ch == "]"):
                stack.pop()
                if not stack:
                    return s[start:i+1]
            else:
                stack.pop()
    return None

def parse_json_strict(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

    cand = _extract_first_json_block(text)
    if cand:
        return json.loads(cand)

    raise GeminiError("Model did not return valid JSON")

async def _gemini_generate_raw(user_prompt: str, max_tokens: int, schema: Optional[Dict[str, Any]]) -> str:
    if not GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY is missing")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    generation: Dict[str, Any] = {
        "temperature": GEMINI_TEMPERATURE,
        "maxOutputTokens": max_tokens,
        "candidateCount": 1,
        "responseMimeType": "application/json",
    }
    # schema 지원/비지원이 섞일 수 있어 400이면 자동 재시도(아래)
    if schema is not None:
        generation["responseSchema"] = schema

    payload: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": generation,
    }

    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json", "User-Agent": USER_AGENT}

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT * 2) as client:
        r = await client.post(url, headers=headers, json=payload)

        if r.status_code >= 400:
            # schema 필드 미지원/검증실패면 1회 schema 없이 재시도
            if schema is not None:
                logger.warning("Gemini schema failed (%s). Retrying without schema.", r.status_code)
                return await _gemini_generate_raw(user_prompt, max_tokens, schema=None)
            raise GeminiError(f"Gemini HTTP {r.status_code}: {r.text[:800]}")

        data = r.json()

    try:
        parts = data["candidates"][0]["content"]["parts"]
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        text = "\n".join([t for t in texts if t]).strip()
        if not text:
            raise KeyError
        return text
    except Exception:
        raise GeminiError("Gemini response parse failed")

async def gemini_generate_json(user_prompt: str, max_tokens: int, schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = await _gemini_generate_raw(user_prompt, max_tokens, schema=schema)
    try:
        return parse_json_strict(raw)
    except Exception:
        # JSON 수리 1회
        repair_prompt = (
            "You MUST output ONLY valid JSON that matches the schema. No markdown. No extra text.\n\n"
            f"INPUT:\n{raw}\n\nOUTPUT: JSON only."
        )
        raw2 = await _gemini_generate_raw(repair_prompt, max_tokens, schema=schema)
        return parse_json_strict(raw2)


# =========================================================
# Post validate sources + question fallback
# =========================================================
def filter_sources_to_allowed(obj: Dict[str, Any], allowed_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    allowed_urls = {s["url"] for s in allowed_sources}
    url_to_title = {s["url"]: s["title"] for s in allowed_sources}

    def clean(lst: Any) -> List[Dict[str, str]]:
        if not isinstance(lst, list):
            return []
        out: List[Dict[str, str]] = []
        for x in lst:
            if not isinstance(x, dict):
                continue
            url = (x.get("url") or "").strip()
            if url in allowed_urls:
                out.append({"title": url_to_title.get(url, x.get("title", "") or ""), "url": url})
        seen = set()
        uniq = []
        for s in out:
            if s["url"] in seen:
                continue
            seen.add(s["url"])
            uniq.append(s)
        return uniq

    if "sources" in obj:
        obj["sources"] = clean(obj["sources"])
    if isinstance(obj.get("blindspot"), dict):
        obj["blindspot"]["sources"] = clean(obj["blindspot"].get("sources", []))
    if "sources_top" in obj:
        obj["sources_top"] = clean(obj["sources_top"])
    if isinstance(obj.get("evidences"), list):
        for ev in obj["evidences"]:
            if isinstance(ev, dict):
                ev["sources"] = clean(ev.get("sources", []))
    return obj

def ensure_three_questions(obj: Dict[str, Any], field: str, conviction: str) -> None:
    qs = obj.get(field)
    if not isinstance(qs, list):
        qs = []
    qs = [str(x) for x in qs if isinstance(x, str) and x.strip()]
    qs = qs[:3]
    while len(qs) < 3:
        if len(qs) == 0:
            qs.append(f"‘{conviction}’가 깨지는 숫자 기준(반증 조건)을 하나 정할 수 있습니까?")
        elif len(qs) == 1:
            qs.append(f"‘{conviction}’를 언제까지 검증할 겁니까? 다음 확인 시점을 날짜/이벤트로 고르세요.")
        else:
            qs.append(f"‘{conviction}’ 전제가 무너질 때 대응 계획은 무엇입니까? (유지/축소/정리 기준)")
    obj[field] = qs


# =========================================================
# Fallback responses (return 200 JSON instead of throwing 502)
# =========================================================
def fallback_short(asset: str, ticker: str, company: str, conviction: str, reason: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "prompt_version": PROMPT_VERSION_SHORT,
        "as_of": now_iso(),
        "asset": asset,
        "ticker": ticker,
        "company": company,
        "conviction_original": conviction,
        "blindspot": {
            "title": "근거 수집 실패",
            "value_line": "근거 수치/팩트가 확인되지 않았습니다.",
            "detail": f"서버가 리포트를 생성하는 중 오류가 발생했습니다. ({reason})\n잠시 후 다시 시도하거나 force_refresh=true로 재시도하세요.",
            "severity": "high",
            "confidence": 0.2,
            "next_metric": "다음 분기 실적/공시에서 확인할 숫자 1개를 지정하세요.",
            "sources": [],
        },
        "questions": [],
        "sources": [],
    }
    ensure_three_questions(out, "questions", conviction)
    return out

def fallback_deep(asset: str, ticker: str, company: str, conviction: str, reason: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "prompt_version": PROMPT_VERSION_DEEP,
        "as_of": now_iso(),
        "asset": asset,
        "ticker": ticker,
        "company": company,
        "conviction_original": conviction,
        "conviction_parsed": {
            "claim": conviction,
            "assumptions": [],
            "time_horizon": "불명",
        },
        "evidences": [],
        "counter_questions": [],
        "sources_top": [],
    }
    ensure_three_questions(out, "counter_questions", conviction)
    # deep 화면이 빈 경우를 대비해 이유를 남기고 싶으면(앱은 무시해도 됨)
    out["__server_note"] = f"deep generation failed: {reason}"
    return out


# =========================================================
# Pipeline
# =========================================================
async def retrieve_context(asset: str, ticker: str, company: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # 키 없으면 검색 스킵
    if SEARCH_PROVIDER == "serper" and not SERPER_API_KEY:
        return [], []
    if SEARCH_PROVIDER == "brave" and not BRAVE_API_KEY:
        return [], []

    queries = build_queries(asset, ticker, company)
    raw_hits = await web_search(queries, per_query=5)
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

    # 근거가 너무 적으면 LLM 호출 자체를 스킵 (비용 절감 + 안정성)
    if len(excerpts) < MIN_EXCERPTS_FOR_LLM:
        out = fallback_short(asset, ticker, company, conviction_norm, f"not enough excerpts ({len(excerpts)})")
        await cache.set(ckey, safe_json_dumps(out), CACHE_TTL_SECONDS)
        return out

    # 프롬프트에는 "간단한 sources + excerpts"만 전달(토큰 절감)
    context_sources_json = safe_json_dumps([
        {"id": s["id"], "title": s["title"], "url": s["url"], "publisher": s["publisher"], "tier": s["tier"]}
        for s in sources
    ])
    context_excerpts_json = safe_json_dumps(excerpts)

    prompt = SHORT_USER_PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION_SHORT,
        asset=asset,
        ticker=ticker,
        company=company,
        conviction=conviction_norm,
        conviction_quality=conviction_quality(conviction_norm),
        now_iso=now_iso(),
        context_sources_json=context_sources_json,
        context_excerpts_json=context_excerpts_json,
    )

    try:
        out = await gemini_generate_json(prompt, GEMINI_MAX_TOKENS_SHORT, schema=SHORT_SCHEMA)
    except Exception as e:
        logger.exception("short LLM failed")
        out = fallback_short(asset, ticker, company, conviction_norm, f"{type(e).__name__}: {str(e)[:200]}")

    # normalize/fill
    out["prompt_version"] = PROMPT_VERSION_SHORT
    out["asset"] = asset
    out["ticker"] = ticker
    out["company"] = company
    out["conviction_original"] = conviction_norm
    out["as_of"] = out.get("as_of") or now_iso()

    out = filter_sources_to_allowed(out, sources)
    ensure_three_questions(out, "questions", conviction_norm)

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

    if len(excerpts) < MIN_EXCERPTS_FOR_LLM:
        out = fallback_deep(asset, ticker, company, conviction_norm, f"not enough excerpts ({len(excerpts)})")
        await cache.set(ckey, safe_json_dumps(out), CACHE_TTL_SECONDS)
        return out

    context_sources_json = safe_json_dumps([
        {"id": s["id"], "title": s["title"], "url": s["url"], "publisher": s["publisher"], "tier": s["tier"]}
        for s in sources
    ])
    context_excerpts_json = safe_json_dumps(excerpts)

    prompt = DEEP_USER_PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION_DEEP,
        asset=asset,
        ticker=ticker,
        company=company,
        conviction=conviction_norm,
        conviction_quality=conviction_quality(conviction_norm),
        now_iso=now_iso(),
        context_sources_json=context_sources_json,
        context_excerpts_json=context_excerpts_json,
    )

    try:
        out = await gemini_generate_json(prompt, GEMINI_MAX_TOKENS_DEEP, schema=DEEP_SCHEMA)
    except Exception as e:
        logger.exception("deep LLM failed")
        out = fallback_deep(asset, ticker, company, conviction_norm, f"{type(e).__name__}: {str(e)[:200]}")

    out["prompt_version"] = PROMPT_VERSION_DEEP
    out["asset"] = asset
    out["ticker"] = ticker
    out["company"] = company
    out["conviction_original"] = conviction_norm
    out["as_of"] = out.get("as_of") or now_iso()

    out = filter_sources_to_allowed(out, sources)
    ensure_three_questions(out, "counter_questions", conviction_norm)

    # evidences safety
    if not isinstance(out.get("evidences"), list):
        out["evidences"] = []
    if len(out["evidences"]) > 5:
        out["evidences"] = out["evidences"][:5]

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
        "model": GEMINI_MODEL,
        "search_provider": SEARCH_PROVIDER,
        "has_gemini_key": bool(GEMINI_API_KEY),
        "has_serper_key": bool(SERPER_API_KEY),
        "has_brave_key": bool(BRAVE_API_KEY),
        "min_excerpts_for_llm": MIN_EXCERPTS_FOR_LLM,
        "time": now_iso(),
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

@app.post("/v1/deep-report")
async def deep_report(req: DeepReportRequest):
    ticker = req.ticker.strip()
    company = (req.company or "Unknown").strip()
    conviction_norm = normalize_conviction(req.conviction)

    if len(conviction_norm) < 10:
        raise HTTPException(status_code=400, detail="conviction must be at least 10 characters")

    asset = infer_asset(req.asset, ticker)
    return await run_deep(asset, ticker, company, conviction_norm, req.force_refresh)

@app.post("/v1/short-report")
async def short_report(req: AnalyzeRequest):
    return await analyze(req)

if __name__ == "__main__":
    import uvicorn
    port = env_int("PORT", 8000)
    uvicorn.run("main:app", host="0.0.0.0", port=port)


