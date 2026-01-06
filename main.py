# main.py
# ============================================================
# SecondLook / THE SHORT — FastAPI Backend (Render-ready)
# - POST /analyze     : 1개 블라인드스팟 + 질문 3개 + 출처
# - POST /deep-report : 반대근거 3~5개 + 질문 3개 + 출처
# - 24h in-memory TTL cache (same payload -> no extra AI cost)
# - Optional Serper search for real URLs (SERPER_API_KEY)
# - Optional Gemini generation (GEMINI_API_KEY)
#
# ✅ No "requests" dependency (uses urllib from stdlib)
# ✅ Won't crash on boot if env vars missing (safe fallback)
# ============================================================

from _future_ import annotations

import os
import re
import json
import time
import hashlib
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Literal, List, Dict, Tuple
from collections import OrderedDict
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------
# Config
# -----------------------------
SERVICE_NAME = "secondlook"
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "A1")

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24h default
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "5000"))

ENABLE_SEARCH = os.getenv("ENABLE_SEARCH", "true").lower() in ("1", "true", "yes", "y")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")  # optional

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # optional
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.0-flash")  # user preference
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.4"))

DEFAULT_LOCALE = os.getenv("DEFAULT_LOCALE", "ko-KR")


# KST timezone (Korea Standard Time)
KST = timezone(timedelta(hours=9))


# -----------------------------
# TTL Cache (fixes your error)
# -----------------------------
class TTLMemoryCache:
    """
    - ttl_seconds 동안 캐시 유지
    - max_items 초과하면 오래된 것부터 제거
    - in-memory라 Render free 인스턴스 스핀다운되면 캐시 증발(정상)
    """
    def _init_(self, ttl_seconds: int = 86400, max_items: int = 5000):
        self.ttl_seconds = int(ttl_seconds)
        self.max_items = int(max_items)
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            exp, val = item
            if exp < now:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return val

    def set(self, key: str, value: Any) -> None:
        exp = time.time() + self.ttl_seconds
        with self._lock:
            if key in self._data:
                self._data.pop(key, None)
            self._data[key] = (exp, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ttl_seconds": self.ttl_seconds,
                "max_items": self.max_items,
                "items": len(self._data),
            }


cache = TTLMemoryCache(ttl_seconds=CACHE_TTL_SECONDS, max_items=CACHE_MAX_ITEMS)


# -----------------------------
# Models
# -----------------------------
Asset = Literal["US", "KR", "COIN"]

class AnalyzePayload(BaseModel):
    asset: Asset
    ticker: str = Field(min_length=1, max_length=32)
    company: str = Field(default="", max_length=128)
    conviction: str = Field(min_length=3, max_length=500)
    locale: str = Field(default=DEFAULT_LOCALE, max_length=16)
    forceRefresh: bool = False


class SourceItem(BaseModel):
    title: str
    url: str
    publisher: Optional[str] = None
    snippet: Optional[str] = None


class BlindspotOut(BaseModel):
    severity: Literal["low", "medium", "high"]
    title: str
    valueLine: str
    detail: str


class AnalyzeResponse(BaseModel):
    ok: bool
    cached: bool
    asOfISO: str
    asOfKorean: str
    selection: Dict[str, str]
    conviction: str
    blindspot: BlindspotOut
    questions: List[str]
    sources: List[SourceItem]
    disclaimer: str


class CounterEvidence(BaseModel):
    severity: Literal["low", "medium", "high"]
    title: str
    fact: str
    whyItMatters: str
    sources: List[SourceItem]


class DeepReportResponse(BaseModel):
    ok: bool
    cached: bool
    asOfISO: str
    asOfKorean: str
    selection: Dict[str, str]
    conviction: str
    summary: str
    counterEvidence: List[CounterEvidence]
    questions: List[str]
    sources: List[SourceItem]
    disclaimer: str


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="SecondLook API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # iOS app 호출 편의
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Helpers: date formatting
# -----------------------------
def now_kst() -> datetime:
    return datetime.now(tz=KST)

def iso_kst(dt: datetime) -> str:
    return dt.isoformat()

def korean_datetime(dt: datetime) -> str:
    # 예: 2026년 1월 6일 오전 8:54
    hour = dt.hour
    ampm = "오전" if hour < 12 else "오후"
    h12 = hour % 12
    if h12 == 0:
        h12 = 12
    return f"{dt.year}년 {dt.month}월 {dt.day}일 {ampm} {h12}:{dt.minute:02d}"


# -----------------------------
# Helpers: cache key
# -----------------------------
def _normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def make_cache_key(endpoint: str, payload: AnalyzePayload) -> str:
    base = {
        "endpoint": endpoint,
        "prompt_version": PROMPT_VERSION,
        "asset": payload.asset,
        "ticker": _normalize_text(payload.ticker).upper(),
        "company": _normalize_text(payload.company),
        "conviction": _normalize_text(payload.conviction),
        "locale": payload.locale,
        "model": GEMINI_MODEL,
        "temp": GEMINI_TEMPERATURE,
        "search": bool(ENABLE_SEARCH and SERPER_API_KEY),
    }
    raw = json.dumps(base, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# -----------------------------
# HTTP JSON (stdlib)
# -----------------------------
def http_post_json(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, headers={**headers, "Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
            return json.loads(payload)
    except HTTPError as e:
        detail = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"HTTPError {e.code}: {detail}")
    except URLError as e:
        raise RuntimeError(f"URLError: {e}")
    except Exception as e:
        raise RuntimeError(str(e))


# -----------------------------
# Search (Serper optional)
# -----------------------------
def serper_search(query: str, k: int = 6) -> List[SourceItem]:
    """
    Serper Google Search API
    - env: SERPER_API_KEY
    - returns top organic results (title, link, snippet)
    """
    if not (ENABLE_SEARCH and SERPER_API_KEY):
        return []

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
    body = {"q": query, "num": k}

    out = http_post_json(url, headers=headers, body=body, timeout=20)
    items: List[SourceItem] = []

    organic = out.get("organic", []) or []
    for r in organic[:k]:
        title = str(r.get("title") or "").strip()
        link = str(r.get("link") or "").strip()
        snippet = str(r.get("snippet") or "").strip()
        if title and link:
            items.append(SourceItem(title=title, url=link, snippet=snippet))

    return items


def source_fallback_candidates(asset: Asset, ticker: str, company: str) -> List[SourceItem]:
    """
    Serper 없을 때라도 UI에 '출처 후보'를 보여줄 수 있게 하는 안전한 기본 링크들.
    (구체 숫자/사실은 여기서 검증 불가 → LLM에도 "숫자 만들지 말라" 강제)
    """
    t = ticker.upper().strip()
    c = company.strip() or "Unknown"

    if asset == "US":
        return [
            SourceItem(title="SEC EDGAR Search", url="https://www.sec.gov/edgar/search/"),
            SourceItem(title=f"{t} Investor Relations (검색)", url=f"https://www.google.com/search?q={t}+investor+relations"),
            SourceItem(title="NASDAQ Market Activity", url="https://www.nasdaq.com/market-activity"),
        ]
    if asset == "KR":
        return [
            SourceItem(title="DART 전자공시", url="https://dart.fss.or.kr/"),
            SourceItem(title="KRX 정보데이터시스템", url="https://data.krx.co.kr/"),
            SourceItem(title=f"{c} 뉴스/공시(검색)", url=f"https://www.google.com/search?q={c}+공시+실적"),
        ]
    # COIN
    return [
        SourceItem(title="CoinMarketCap", url="https://coinmarketcap.com/"),
        SourceItem(title="CoinGecko", url="https://www.coingecko.com/"),
        SourceItem(title="TokenUnlocks", url="https://token.unlocks.app/"),
    ]


def build_search_queries(asset: Asset, ticker: str, company: str) -> List[str]:
    t = ticker.strip()
    c = company.strip()

    if asset == "US":
        return [
            f"{t} {c} earnings risk margin competition regulation",
            f"{t} {c} valuation multiple downside bear case",
        ]
    if asset == "KR":
        # 한국어 쿼리 섞기
        base = c if c else t
        return [
            f"{base} 실적 리스크 마진 경쟁 규제 악재",
            f"{base} 밸류에이션 고평가 저평가 논란 PER PBR",
        ]
    # COIN
    return [
        f"{t} token unlock schedule inflation circulating supply",
        f"{t} regulatory risk investigation fine ban",
    ]


# -----------------------------
# Gemini (optional)
# -----------------------------
def gemini_ready() -> bool:
    return bool(GEMINI_API_KEY)

def call_gemini_json(system: str, user: str) -> Optional[Dict[str, Any]]:
    """
    Returns dict parsed from model output (must be JSON).
    If Gemini not configured or output invalid -> None.
    """
    if not GEMINI_API_KEY:
        return None

    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=GEMINI_API_KEY)

        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system,
        )

        resp = model.generate_content(
            user,
            generation_config={
                "temperature": GEMINI_TEMPERATURE,
            },
        )

        text = getattr(resp, "text", None) or ""
        text = text.strip()

        # Extract JSON (model sometimes wraps with fences)
        # Find first '{' and last '}'.
        i = text.find("{")
        j = text.rfind("}")
        if i == -1 or j == -1 or j <= i:
            return None
        js = text[i : j + 1]

        return json.loads(js)

    except Exception:
        # Don't crash server on AI error
        return None


# -----------------------------
# Prompt slots (A1)
# -----------------------------
def system_prompt_common() -> str:
    return (
        "너는 '더쇼트(The Short)'의 반대근거 엔진이다.\n"
        "목표: 사용자의 매수/보유 논리를 '팩트와 리스크'로 검증한다.\n"
        "톤: 차갑고 단정적. 하지만 욕설/비하/인신공격 금지. 사람을 치지 말고 논리를 친다.\n"
        "팩트 규칙:\n"
        "- 숫자를 말할 때는 주어진 SOURCES 안에서 근거가 보일 때만 사용한다.\n"
        "- 근거가 불충분하면 숫자를 꾸미지 말고 '확인 필요'로 처리한다.\n"
        "- 출처 URL은 반드시 SOURCES에 있는 링크만 사용한다(새로 만들어내지 마라).\n"
        "출력 규칙:\n"
        "- 반드시 '오직 JSON'만 출력한다. 설명 텍스트, 마크다운, 코드블록 금지.\n"
    )

def user_prompt_analyze(payload: AnalyzePayload, sources: List[SourceItem]) -> str:
    sel = {"asset": payload.asset, "ticker": payload.ticker, "company": payload.company}
    return (
        f"INPUT\n"
        f"- selection: {json.dumps(sel, ensure_ascii=False)}\n"
        f"- conviction: {payload.conviction}\n"
        f"- locale: {payload.locale}\n\n"
        f"SOURCES (use only these URLs)\n"
        f"{json.dumps([s.model_dump() for s in sources], ensure_ascii=False)}\n\n"
        f"OUTPUT JSON SCHEMA\n"
        f"{{\n"
        f'  "blindspot": {{"severity":"low|medium|high","title":"...","valueLine":"...","detail":"..."}},\n'
        f'  "questions": ["...","...","..."],\n'
        f'  "sources": [{{"title":"...","url":"...","publisher":"optional","snippet":"optional"}}]\n'
        f"}}\n\n"
        f"RULES\n"
        f"- blindspot은 1개만. 가장 치명적인 전제 누락/리스크를 한 방으로.\n"
        f"- valueLine은 '한 줄 팩트/리스크'로 짧고 강하게.\n"
        f"- questions는 3개. 회피 못 하게 '조건/수치/손절/검증지표'를 묻는다.\n"
        f"- sources는 최대 3개. 반드시 위 SOURCES 안에서만 선택.\n"
    )

def user_prompt_deep(payload: AnalyzePayload, sources: List[SourceItem]) -> str:
    sel = {"asset": payload.asset, "ticker": payload.ticker, "company": payload.company}
    return (
        f"INPUT\n"
        f"- selection: {json.dumps(sel, ensure_ascii=False)}\n"
        f"- conviction: {payload.conviction}\n"
        f"- locale: {payload.locale}\n\n"
        f"SOURCES (use only these URLs)\n"
        f"{json.dumps([s.model_dump() for s in sources], ensure_ascii=False)}\n\n"
        f"OUTPUT JSON SCHEMA\n"
        f"{{\n"
        f'  "summary": "...",\n'
        f'  "counterEvidence": [\n'
        f'    {{"severity":"low|medium|high","title":"...","fact":"...","whyItMatters":"...","sources":[{{"title":"...","url":"...","publisher":"optional","snippet":"optional"}}]}},\n'
        f'    ... (3~5 items)\n'
        f'  ],\n'
        f'  "questions": ["...","...","..."],\n'
        f'  "sources": [{{"title":"...","url":"...","publisher":"optional","snippet":"optional"}}]\n'
        f"}}\n\n"
        f"RULES\n"
        f"- counterEvidence는 3~5개. 서로 다른 축(수요/마진/경쟁/규제/밸류/공급)을 섞어라.\n"
        f"- fact는 '짧고 단정적'. 숫자가 필요하면 SOURCES snippet에 근거가 있을 때만.\n"
        f"- whyItMatters는 1~2문장. 이게 왜 가격/리스크로 이어지는지.\n"
        f"- sources는 각 evidence당 1~2개. 전체 sources는 최대 5개.\n"
        f"- URL은 반드시 SOURCES에서만.\n"
    )


# -----------------------------
# Fallback (no Gemini / parse fail)
# -----------------------------
def heuristic_blindspot(conviction: str) -> BlindspotOut:
    low = conviction.lower()
    if any(k in low for k in ["매출", "성장", "수요", "users", "사용자", "daU".lower()]):
        return BlindspotOut(
            severity="high",
            title="성장 서사가 '이익/현금흐름'으로 연결되는 구간이 비어있음",
            valueLine="매출↑는 주가↑가 아니다. 마진/현금흐름이 꺾이면 시나리오는 끝난다.",
            detail="성장은 비용·가격경쟁·경기·규제 앞에서 먼저 흔들립니다. '언제/어떤 지표로 확인할지'가 없으면 확신이 아니라 추정입니다.",
        )
    if any(k in low for k in ["저평가", "밸류", "per", "p/e", "pbr", "undervalued"]):
        return BlindspotOut(
            severity="medium",
            title="저평가가 아니라 '정당한 할인'일 수 있음",
            valueLine="싸서 오르는 게 아니라, 싫어할 이유가 사라져야 오른다.",
            detail="멀티플은 자동복구가 아닙니다. 경쟁/규제/수익성 같은 할인 요인이 유지되면 평균회귀는 오지 않습니다.",
        )
    if any(k in low for k in ["독점", "moat", "해자", "점유율", "경쟁"]):
        return BlindspotOut(
            severity="high",
            title="해자의 본질은 점유율이 아니라 '가격 결정력'",
            valueLine="가격을 못 지키면 해자는 무너진다.",
            detail="경쟁이 가격으로 들어오면 점유율은 남아도 수익성이 먼저 깨집니다. '가격을 올릴 수 있는 이유'가 문장에 없습니다.",
        )
    return BlindspotOut(
        severity="medium",
        title="핵심 전제가 문장에 없음",
        valueLine="'왜 지금'과 '언제까지'가 비어있다.",
        detail="좋은 논리는 시간과 조건을 포함합니다. 지금 문장은 신념에 가깝고, 검증 가능한 조건이 약합니다.",
    )

def heuristic_questions() -> List[str]:
    return [
        "이 논리가 깨지는 '조건' 1개를 말할 수 있습니까?",
        "손절 기준(가격/지표)은 무엇입니까? '없다'는 답도 하나의 답입니다.",
        "다음 분기에 반드시 확인할 '숫자 1개'를 정할 수 있습니까?",
    ]

def disclaimer_text() -> str:
    return "투자 조언이 아닙니다. 정보 제공 목적이며 최종 판단과 책임은 사용자에게 있습니다."


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": SERVICE_NAME,
        "version": app.version,
        "prompt_version": PROMPT_VERSION,
        "cache": cache.stats(),
        "search_enabled": bool(ENABLE_SEARCH and SERPER_API_KEY),
        "gemini_ready": gemini_ready(),
        "model": GEMINI_MODEL,
    }

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzePayload):
    # cache
    key = make_cache_key("analyze", payload)
    if not payload.forceRefresh:
        hit = cache.get(key)
        if hit is not None:
            return AnalyzeResponse(**hit, cached=True)

    # sources (search if possible)
    sources: List[SourceItem] = []
    if ENABLE_SEARCH and SERPER_API_KEY:
        queries = build_search_queries(payload.asset, payload.ticker, payload.company)
        # cheap: 1 query만 사용(비용/속도)
        sources = serper_search(queries[0], k=6)
    if not sources:
        sources = source_fallback_candidates(payload.asset, payload.ticker, payload.company)

    # Gemini
    system = system_prompt_common()
    user = user_prompt_analyze(payload, sources)

    js = call_gemini_json(system=system, user=user)

    # build response
    dt = now_kst()
    base = {
        "ok": True,
        "cached": False,
        "asOfISO": iso_kst(dt),
        "asOfKorean": korean_datetime(dt),
        "selection": {"asset": payload.asset, "ticker": payload.ticker, "company": payload.company or "Unknown"},
        "conviction": _normalize_text(payload.conviction),
        "disclaimer": disclaimer_text(),
    }

    if isinstance(js, dict):
        try:
            blind = BlindspotOut(**js.get("blindspot", {}))
            questions = js.get("questions") or []
            if not isinstance(questions, list) or len(questions) < 3:
                questions = heuristic_questions()
            else:
                questions = [str(x) for x in questions[:3]]

            out_sources = js.get("sources") or []
            if not isinstance(out_sources, list) or len(out_sources) == 0:
                out_sources = [s.model_dump() for s in sources[:3]]
            else:
                # safety: keep only URLs from allowed sources
                allowed = {s.url for s in sources}
                filtered = []
                for it in out_sources:
                    if not isinstance(it, dict):
                        continue
                    url = str(it.get("url") or "").strip()
                    title = str(it.get("title") or "").strip()
                    if url in allowed and title:
                        filtered.append(it)
                if not filtered:
                    filtered = [s.model_dump() for s in sources[:3]]
                out_sources = filtered[:3]

            resp = {**base, "blindspot": blind.model_dump(), "questions": questions, "sources": out_sources}
            cache.set(key, resp)
            return AnalyzeResponse(**resp)
        except Exception:
            # fallthrough to heuristic
            pass

    # fallback
    blind = heuristic_blindspot(payload.conviction)
    resp = {
        **base,
        "blindspot": blind.model_dump(),
        "questions": heuristic_questions(),
        "sources": [s.model_dump() for s in sources[:3]],
    }
    cache.set(key, resp)
    return AnalyzeResponse(**resp)


@app.post("/deep-report", response_model=DeepReportResponse)
def deep_report(payload: AnalyzePayload):
    key = make_cache_key("deep-report", payload)
    if not payload.forceRefresh:
        hit = cache.get(key)
        if hit is not None:
            return DeepReportResponse(**hit, cached=True)

    # sources (deep-report는 2개 쿼리까지)
    sources: List[SourceItem] = []
    if ENABLE_SEARCH and SERPER_API_KEY:
        queries = build_search_queries(payload.asset, payload.ticker, payload.company)
        merged: Dict[str, SourceItem] = {}
        for q in queries[:2]:
            for s in serper_search(q, k=6):
                merged[s.url] = s
        sources = list(merged.values())[:10]

    if not sources:
        sources = source_fallback_candidates(payload.asset, payload.ticker, payload.company)

    system = system_prompt_common()
    user = user_prompt_deep(payload, sources)

    js = call_gemini_json(system=system, user=user)

    dt = now_kst()
    base = {
        "ok": True,
        "cached": False,
        "asOfISO": iso_kst(dt),
        "asOfKorean": korean_datetime(dt),
        "selection": {"asset": payload.asset, "ticker": payload.ticker, "company": payload.company or "Unknown"},
        "conviction": _normalize_text(payload.conviction),
        "disclaimer": disclaimer_text(),
    }

    if isinstance(js, dict):
        try:
            summary = str(js.get("summary") or "").strip()
            if not summary:
                summary = "논리의 취약점이 확인됐습니다. 핵심은 '검증 가능한 조건'과 '리스크 관리'입니다."

            ce_raw = js.get("counterEvidence") or []
            if not isinstance(ce_raw, list) or len(ce_raw) < 3:
                raise ValueError("counterEvidence too small")

            # safety: only allow URLs from provided sources
            allowed_urls = {s.url for s in sources}

            counter: List[Dict[str, Any]] = []
            all_srcs: Dict[str, SourceItem] = {}

            for item in ce_raw[:5]:
                if not isinstance(item, dict):
                    continue
                sev = str(item.get("severity") or "medium").strip()
                if sev not in ("low", "medium", "high"):
                    sev = "medium"

                title = str(item.get("title") or "").strip()
                fact = str(item.get("fact") or "").strip()
                why = str(item.get("whyItMatters") or "").strip()

                srcs = []
                for s in (item.get("sources") or [])[:2]:
                    if not isinstance(s, dict):
                        continue
                    url = str(s.get("url") or "").strip()
                    st = str(s.get("title") or "").strip()
                    if url in allowed_urls and st:
                        srcs.append({"title": st, "url": url, "publisher": s.get("publisher"), "snippet": s.get("snippet")})
                        all_srcs[url] = SourceItem(title=st, url=url, publisher=s.get("publisher"), snippet=s.get("snippet"))

                if title and fact and why:
                    if not srcs:
                        # if model forgot sources, attach one from candidates (still allowed)
                        fallback_src = sources[0]
                        srcs = [fallback_src.model_dump()]
                        all_srcs[fallback_src.url] = fallback_src

                    counter.append({
                        "severity": sev,
                        "title": title,
                        "fact": fact,
                        "whyItMatters": why,
                        "sources": srcs,
                    })

            if len(counter) < 3:
                raise ValueError("counterEvidence filtered too much")

            questions = js.get("questions") or []
            if not isinstance(questions, list) or len(questions) < 3:
                questions = heuristic_questions()
            else:
                questions = [str(x) for x in questions[:3]]

            # top sources for page footer
            footer_sources = list(all_srcs.values())
            if not footer_sources:
                footer_sources = sources[:5]

            resp = {
                **base,
                "summary": summary,
                "counterEvidence": counter,
                "questions": questions,
                "sources": [s.model_dump() for s in footer_sources[:5]],
            }
            cache.set(key, resp)
            return DeepReportResponse(**resp)

        except Exception:
            pass

    # fallback deep-report (no Gemini or parse fail)
    # produce non-empty content so your DeepReport screen isn't blank
    b = heuristic_blindspot(payload.conviction)
    ev = [
        CounterEvidence(
            severity="high" if b.severity == "high" else "medium",
            title="전제 누락: 검증 조건이 없음",
            fact=b.valueLine,
            whyItMatters="조건이 없으면 틀렸을 때도 계속 들고 갑니다. 리스크는 '확률'이 아니라 '대응'에서 터집니다.",
            sources=[s for s in sources[:1]],
        ).model_dump(),
        CounterEvidence(
            severity="medium",
            title="리스크 관리: 손절/무효화 기준 부재",
            fact="손절 기준이 없으면 변동성이 '손실'로 고정됩니다.",
            whyItMatters="시장에선 '맞았냐'보다 '틀렸을 때 얼마나 잃었냐'가 성과를 결정합니다.",
            sources=[s for s in sources[1:2]] if len(sources) > 1 else [sources[0].model_dump()],
        ).model_dump(),
        CounterEvidence(
            severity="medium",
            title="확인 지표: 다음 분기 체크 포인트 미정",
            fact="다음 분기에 확인할 숫자 1개가 없으면 학습이 안 됩니다.",
            whyItMatters="확신을 업데이트하지 못하면 반복 손실로 갑니다. 체크 포인트가 곧 방어력입니다.",
            sources=[s for s in sources[2:3]] if len(sources) > 2 else [sources[0].model_dump()],
        ).model_dump(),
    ]

    resp = {
        **base,
        "summary": "심층 검증: 논리를 '조건/지표/리스크 관리'로 분해했습니다.",
        "counterEvidence": ev,
        "questions": heuristic_questions(),
        "sources": [s.model_dump() for s in sources[:5]],
    }
    cache.set(key, resp)
    return DeepReportResponse(**resp)
