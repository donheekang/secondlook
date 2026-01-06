import os
import json
import time
import hashlib
import logging
import threading
from datetime import datetime, timezone
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================================================
# THE SHORT Backend (Render-friendly)
# - FastAPI + httpx
# - 24h in-memory TTL cache
# - Serper search (optional) for sources
# - Gemini generate (optional) for AI
# - Never crash on startup because of missing deps/keys
# =========================================================

# -----------------------
# Logging
# -----------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("the-short")


# -----------------------
# TTL Memory Cache (FIXED: accepts args)
# -----------------------
class TTLMemoryCache:
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


CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24h default
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "5000"))
cache = TTLMemoryCache(ttl_seconds=CACHE_TTL_SECONDS, max_items=CACHE_MAX_ITEMS)


# -----------------------
# Env / Config
# -----------------------
# iOS 앱 ↔️ 서버 간 간단 인증키 (선택)
SERVER_API_KEY = os.getenv("SERVER_API_KEY", "").strip()
# Serper (구글 검색 API) - 출처 리스트 만들 때 사용 (선택)
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()

# Gemini API key (필수로 쓰고 싶으면 넣어야 함)
# 구글 키 이름이 GOOGLE_API_KEY 인 경우도 있어서 둘 다 지원
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")).strip()

# 모델명은 네가 Render env로 바꿀 수 있게 열어둠
GEMINI_MODEL_ANALYZE = os.getenv("GEMINI_MODEL_ANALYZE", os.getenv("GEMINI_MODEL", "gemini-3.0-flash")).strip()
GEMINI_MODEL_DEEP = os.getenv("GEMINI_MODEL_DEEP", os.getenv("GEMINI_MODEL", "gemini-3.0-flash")).strip()

# Generative Language API (기본은 v1beta)
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").strip()

SERPER_ENDPOINT = os.getenv("SERPER_ENDPOINT", "https://google.serper.dev/search").strip()

# CORS (웹 디버깅용; iOS 앱은 보통 필요 없지만 열어둠)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").strip()


# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(title="The Short API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[""] if ALLOWED_ORIGINS == "" else [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_http: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def _startup() -> None:
    global _http
    _http = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={"User-Agent": "the-short/1.0"},
    )
    logger.info("startup ok")


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _http
    if _http:
        await _http.aclose()
    _http = None
    logger.info("shutdown ok")


# -----------------------
# Models
# -----------------------
AssetType = Literal["US", "KR", "COIN"]
SeverityType = Literal["LOW", "MED", "HIGH"]


class AnalyzePayload(BaseModel):
    asset: AssetType = Field(..., description="US | KR | COIN")
    ticker: str = Field(..., min_length=1, max_length=20)
    company: Optional[str] = Field(None, max_length=80)
    conviction: str = Field(..., min_length=3, max_length=300)
    locale: str = Field("ko-KR")
    forceRefresh: bool = Field(False)


class SourceItem(BaseModel):
    title: str
    url: str
    domain: str


class Blindspot(BaseModel):
    title: str
    valueLine: str
    detail: str
    severity: SeverityType


class AnalyzeResponse(BaseModel):
    asOf: str
    cached: bool
    asset: AssetType
    ticker: str
    company: Optional[str]
    conviction: str
    blindspot: Blindspot
    questions: List[str]
    sources: List[SourceItem]


class DeepPayload(BaseModel):
    asset: AssetType
    ticker: str
    company: Optional[str] = None
    conviction: str
    locale: str = "ko-KR"
    forceRefresh: bool = False


class CounterItem(BaseModel):
    category: str
    headline: str
    evidence: str
    numbers: List[Dict[str, str]] = []
    source_refs: List[int] = []  # sources index 참조


class DeepReportResponse(BaseModel):
    asOf: str
    cached: bool
    asset: AssetType
    ticker: str
    company: Optional[str]
    conviction: str
    summary: str
    counters: List[CounterItem]
    questions: List[str]
    sources: List[SourceItem]


# -----------------------
# Helpers
# -----------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(s: str, max_len: int = 400) -> str:
    s = (s or "").strip()
    s = " ".join(s.split())
    return s[:max_len]


def safe_domain(url: str) -> str:
    try:
        u = httpx.URL(url)
        return (u.host or "").lower()
    except Exception:
        return ""


def make_cache_key(prefix: str, payload_dict: Dict[str, Any]) -> str:
    # forceRefresh는 캐시 키에서 제외 (동일 입력이면 같은 키)
    d = {k: v for k, v in payload_dict.items() if k != "forceRefresh"}
    raw = json.dumps(d, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256((prefix + "|" + raw).encode("utf-8")).hexdigest()
    return h


def extract_first_json(text: str) -> Dict[str, Any]:
    # Gemini가 ⁠ json  ⁠으로 감싸도 그냥 첫 { ... }만 뽑아 파싱
    if not text:
        raise ValueError("empty")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no json object")
    snippet = text[start : end + 1]
    return json.loads(snippet)


def require_server_key(x_api_key: Optional[str]) -> None:
    # SERVER_API_KEY를 설정한 경우에만 체크 (설정 안 하면 오픈)
    if not SERVER_API_KEY:
        return
    if not x_api_key or x_api_key.strip() != SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")


# -----------------------
# Serper Search (optional)
# -----------------------
def build_search_query(asset: AssetType, ticker: str, company: Optional[str], conviction: str) -> str:
    base = ticker
    if company and company.lower() != "unknown":
        base += f" {company}"

    conviction = normalize_text(conviction, 120)

    if asset == "COIN":
        # 코인은 이 키워드들이 출처 뽑기에 잘 먹힘
        return f"{base} token unlock schedule circulating supply FDV {conviction}"
    if asset == "KR":
        return f"{base} 실적 리스크 경쟁 규제 밸류에이션 {conviction}"
    # US
    return f"{base} earnings risk competition regulation valuation {conviction}"


def locale_to_serper(locale: str, asset: AssetType) -> Tuple[str, str]:
    # hl: language, gl: country
    # 한국 사용자 기준으로 hl=ko 유지하되, US는 gl=us로 소스 폭을 넓힘
    hl = "ko" if locale.lower().startswith("ko") else "en"
    if asset == "US":
        return hl, "us"
    if asset == "KR":
        return hl, "kr"
    return hl, "us"


async def serper_search_sources(query: str, hl: str, gl: str, num: int = 5) -> List[SourceItem]:
    if not SERPER_API_KEY:
        return []
    if not _http:
        return []

    try:
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        body = {"q": query, "hl": hl, "gl": gl, "num": num}
        r = await _http.post(SERPER_ENDPOINT, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        organic = data.get("organic", []) or []
        out: List[SourceItem] = []
        for it in organic[:num]:
            url = it.get("link") or ""
            title = it.get("title") or ""
            if not url or not title:
                continue
            out.append(SourceItem(title=title[:140], url=url, domain=safe_domain(url)))
        return out
    except Exception as e:
        logger.warning(f"serper failed: {e}")
        return []


# -----------------------
# Gemini (optional)
# -----------------------
def system_prompt_ko() -> str:
    # 말투: 차갑고 단정. 욕설/비하 금지.
    return (
        "너는 '더 쇼트(The Short)'의 데이터 검증관이다.\n"
        "- 목표: 사용자의 투자 논리에서 가장 취약한 전제 1개를 '팩트 중심'으로 찌른다.\n"
        "- 톤: 차갑고 단정, 짧고 강하다. 하지만 욕설/비하/혐오 표현은 금지.\n"
        "- 출처가 제공되면, 그 범위 내에서만 추론하고 과장은 하지 마라.\n"
        "- 결과는 반드시 JSON만 출력한다(마크다운/설명 금지).\n"
    )


def build_analyze_prompt(payload: AnalyzePayload, sources: List[SourceItem]) -> str:
    src_lines = []
    for i, s in enumerate(sources):
        src_lines.append(f"[{i}] {s.title} ({s.domain}) {s.url}")

    src_block = "\n".join(src_lines) if src_lines else "(no sources provided)"

    return (
        f"{system_prompt_ko()}\n"
        "다음 입력을 분석하라.\n"
        f"- asset: {payload.asset}\n"
        f"- ticker: {payload.ticker}\n"
        f"- company: {payload.company or 'Unknown'}\n"
        f"- conviction: {payload.conviction}\n\n"
        "가능하면 아래 '출처 목록'에서 근거 힌트를 사용하라.\n"
        f"출처 목록:\n{src_block}\n\n"
        "아래 스키마로 JSON만 출력:\n"
        "{\n"
        '  "blindspot": {\n'
        '    "title": "짧은 제목",\n'
        '    "valueLine": "강한 한 줄 팩트/경고",\n'
        '    "detail": "2~4문장 설명(과장 금지)",\n'
        '    "severity": "LOW|MED|HIGH"\n'
        "  },\n"
        '  "questions": ["질문1", "질문2", "질문3"]\n'
        "}\n"
    )


def build_deep_prompt(payload: DeepPayload, sources: List[SourceItem]) -> str:
    src_lines = []
    for i, s in enumerate(sources):
        src_lines.append(f"[{i}] {s.title} ({s.domain}) {s.url}")
    src_block = "\n".join(src_lines) if src_lines else "(no sources provided)"

    return (
        f"{system_prompt_ko()}\n"
        "다음 입력을 기반으로 '심층 반격 리포트'를 작성하라.\n"
        "- 요구사항:\n"
        "  1) 반대 근거 3~5개 (수요/마진/경쟁/규제/밸류에이션 중 최소 3범주)\n"
        "  2) 각 근거는 headline(짧게) + evidence(2~4문장) + numbers(있으면) + source_refs(출처 index)\n"
        "  3) 마지막에 날카로운 질문 3개\n"
        "  4) 욕설/비하/혐오 금지. 단정하고 냉정.\n\n"
        f"- asset: {payload.asset}\n"
        f"- ticker: {payload.ticker}\n"
        f"- company: {payload.company or 'Unknown'}\n"
        f"- conviction: {payload.conviction}\n\n"
        f"출처 목록:\n{src_block}\n\n"
        "아래 스키마로 JSON만 출력:\n"
        "{\n"
        '  "summary": "2~3문장 요약",\n'
        '  "counters": [\n'
        "    {\n"
        '      "category": "수요|마진|경쟁|규제|밸류에이션",\n'
        '      "headline": "짧고 강한 한 줄",\n'
        '      "evidence": "2~4문장",\n'
        '      "numbers": [{"label":"", "value":""}],\n'
        '      "source_refs": [0,1]\n'
        "    }\n"
        "  ],\n"
        '  "questions": ["질문1", "질문2", "질문3"]\n'
        "}\n"
    )


async def gemini_generate(model: str, prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("missing GEMINI_API_KEY (or GOOGLE_API_KEY)")
    if not _http:
        raise RuntimeError("http client not ready")

    url = f"{GEMINI_BASE_URL}/models/{model}:generateContent"
    params = {"key": GEMINI_API_KEY}

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 900,
        },
    }

    r = await _http.post(url, params=params, json=body)
    if r.status_code >= 400:
        raise RuntimeError(f"gemini http {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise RuntimeError("gemini response parse failed")


# -----------------------
# Fallback (no AI / AI failed)
# -----------------------
def fallback_short(payload: AnalyzePayload, sources: List[SourceItem]) -> AnalyzeResponse:
    c = normalize_text(payload.conviction, 220).lower()
    if any(k in c for k in ["매출", "성장", "수요"]):
        blind = Blindspot(
            title="수요 성장 = 이익 성장? (가정 누락)",
            valueLine="매출↑는 이익↑가 아닙니다. 마진이 꺾이면 시나리오는 끝납니다.",
            detail="성장 스토리는 비용/가격/경쟁 앞에서 먼저 흔들립니다. ‘얼마나 벌 수 있는가’가 빠지면 확신은 방어력이 없습니다.",
            severity="HIGH",
        )
    elif any(k in c for k in ["저평가", "밸류", "per", "p/e"]):
        blind = Blindspot(
            title="저평가가 아니라 ‘정당한 할인’",
            valueLine="싸서 오르는 게 아니라, ‘싫어할 이유’가 사라져야 오릅니다.",
            detail="멀티플은 선물이 아닙니다. 할인 요인이 유지되면 평균회귀는 오지 않습니다.",
            severity="MED",
        )
    elif any(k in c for k in ["독점", "경쟁", "moat", "점유율"]):
        blind = Blindspot(
            title="해자(Moat)의 본질",
            valueLine="점유율이 아니라 ‘가격 결정력’이 무너지면 끝입니다.",
            detail="경쟁이 가격으로 들어오면 해자는 생각보다 빨리 무너집니다. ‘가격을 지킬 이유’가 문장에 없습니다.",
            severity="HIGH",
        )
    else:
        blind = Blindspot(
            title="핵심 전제가 문장에 없음",
            valueLine="‘왜 지금’과 ‘언제까지’가 비어있습니다.",
            detail="좋은 논리는 시간/조건을 포함합니다. 지금 문장은 신념에 가깝고, 검증 가능한 조건이 적습니다.",
            severity="MED",
        )

    questions = [
        "이 논리가 깨지는 조건 1가지는 무엇입니까?",
        "실적이 좋아도 주가가 떨어질 수 있는 이유를 말할 수 있습니까?",
        "다음 분기에 반드시 확인할 ‘숫자’ 1개를 정할 수 있습니까?",
    ]

    return AnalyzeResponse(
        asOf=now_iso(),
        cached=False,
        asset=payload.asset,
        ticker=payload.ticker,
        company=payload.company,
        conviction=payload.conviction,
        blindspot=blind,
        questions=questions,
        sources=sources,
    )


def fallback_deep(payload: DeepPayload, sources: List[SourceItem]) -> DeepReportResponse:
    counters = [
        CounterItem(
            category="수요",
            headline="수요는 ‘좋다’가 아니라 ‘지속’이 핵심입니다.",
            evidence="수요가 늘어도 경기/금리/경쟁 변수로 꺾일 수 있습니다. ‘어떤 지표로 지속을 확인할지’가 빠지면 논리는 취약합니다.",
            numbers=[],
            source_refs=[],
        ),
        CounterItem(
            category="마진",
            headline="매출이 아니라 마진이 먼저 무너집니다.",
            evidence="원가/판관비/가격경쟁이 붙는 순간, 매출 성장의 의미가 바뀝니다. 마진 방어 근거가 없으면 리스크가 커집니다.",
            numbers=[],
            source_refs=[],
        ),
        CounterItem(
            category="밸류에이션",
            headline="싸 보이는 건 이유가 있을 때가 많습니다.",
            evidence="멀티플은 ‘평균’으로 돌아오는 게 아니라, 할인 요인이 제거될 때 움직입니다. 무엇이 해소되면 리레이팅 되는지 정의가 필요합니다.",
            numbers=[],
            source_refs=[],
        ),
    ]

    questions = [
        "당신의 논리가 깨지는 조건 1가지를 문장으로 고정할 수 있습니까?",
        "손절/익절 기준은 가격이 아니라 ‘조건’으로 설명할 수 있습니까?",
        "다음 분기에 확인할 숫자 1개를 못 정하면, 무엇을 믿고 있습니까?",
    ]

    return DeepReportResponse(
        asOf=now_iso(),
        cached=False,
        asset=payload.asset,
        ticker=payload.ticker,
        company=payload.company,
        conviction=payload.conviction,
        summary="논리의 핵심 전제가 ‘조건/지표’로 고정되지 않아 방어력이 낮습니다. 반대 시나리오를 숫자와 출처로 잠그는 단계가 필요합니다.",
        counters=counters,
        questions=questions,
        sources=sources,
    )


# -----------------------
# Error handler (always JSON)
# -----------------------
@app.exception_handler(Exception)
async def _all_exception_handler(request: Request, exc: Exception):
    logger.exception(f"unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "Internal Server Error"},
    )


# -----------------------
# Routes
# -----------------------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "time": now_iso(),
        "cache": cache.stats(),
        "has_SERVER_API_KEY": bool(SERVER_API_KEY),
        "has_SERPER_API_KEY": bool(SERPER_API_KEY),
        "has_GEMINI_API_KEY": bool(GEMINI_API_KEY),
        "models": {"analyze": GEMINI_MODEL_ANALYZE, "deep": GEMINI_MODEL_DEEP},
    }


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzePayload, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    require_server_key(x_api_key)

    payload.ticker = normalize_text(payload.ticker, 20).upper()
    payload.company = normalize_text(payload.company or "Unknown", 80)
    payload.conviction = normalize_text(payload.conviction, 300)

    ck = make_cache_key("analyze", payload.model_dump())
    if not payload.forceRefresh:
        hit = cache.get(ck)
        if hit:
            hit["cached"] = True
            return hit

    # sources (optional)
    query = build_search_query(payload.asset, payload.ticker, payload.company, payload.conviction)
    hl, gl = locale_to_serper(payload.locale, payload.asset)
    sources = await serper_search_sources(query, hl=hl, gl=gl, num=5)

    # AI (optional)
    try:
        if GEMINI_API_KEY:
            prompt = build_analyze_prompt(payload, sources)
            raw = await gemini_generate(GEMINI_MODEL_ANALYZE, prompt)
            data = extract_first_json(raw)

            blind = data.get("blindspot") or {}
            qs = data.get("questions") or []

            resp = AnalyzeResponse(
                asOf=now_iso(),
                cached=False,
                asset=payload.asset,
                ticker=payload.ticker,
                company=payload.company,
                conviction=payload.conviction,
                blindspot=Blindspot(
                    title=normalize_text(blind.get("title", "핵심 전제 점검"), 80),
                    valueLine=normalize_text(blind.get("valueLine", "불편한 사실 1개가 빠져 있습니다."), 120),
                    detail=normalize_text(blind.get("detail", "전제를 조건과 지표로 고정하지 않으면 논리는 쉽게 흔들립니다."), 500),
                    severity=(blind.get("severity") or "MED").replace("MEDIUM", "MED").upper(),  # 방어
                ),
                questions=[normalize_text(q, 140) for q in (qs[:3] if isinstance(qs, list) else [])] or fallback_short(payload, sources).questions,
                sources=sources,
            )

            out = resp.model_dump()
            cache.set(ck, out)
            return out

        # no gemini key → fallback
        resp = fallback_short(payload, sources).model_dump()
        cache.set(ck, resp)
        return resp

    except Exception as e:
        logger.warning(f"analyze ai failed -> fallback: {e}")
        resp = fallback_short(payload, sources).model_dump()
        cache.set(ck, resp)
        return resp


@app.post("/v1/deep-report", response_model=DeepReportResponse)
async def deep_report(payload: DeepPayload, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    require_server_key(x_api_key)

    payload.ticker = normalize_text(payload.ticker, 20).upper()
    payload.company = normalize_text(payload.company or "Unknown", 80)
    payload.conviction = normalize_text(payload.conviction, 300)

    ck = make_cache_key("deep", payload.model_dump())
    if not payload.forceRefresh:
        hit = cache.get(ck)
        if hit:
            hit["cached"] = True
            return hit

    query = build_search_query(payload.asset, payload.ticker, payload.company, payload.conviction)
    hl, gl = locale_to_serper(payload.locale, payload.asset)
    sources = await serper_search_sources(query, hl=hl, gl=gl, num=6)

    try:
        if GEMINI_API_KEY:
            prompt = build_deep_prompt(payload, sources)
            raw = await gemini_generate(GEMINI_MODEL_DEEP, prompt)
            data = extract_first_json(raw)

            counters_raw = data.get("counters") or []
            questions_raw = data.get("questions") or []
            summary = normalize_text(data.get("summary", ""), 300) or "반대 근거를 3~5개 축으로 잠급니다. 전제는 숫자와 출처로 고정되어야 합니다."

            counters: List[CounterItem] = []
            if isinstance(counters_raw, list):
                for it in counters_raw[:5]:
                    if not isinstance(it, dict):
                        continue
                    counters.append(
                        CounterItem(
                            category=normalize_text(str(it.get("category", "리스크")), 20),
                            headline=normalize_text(str(it.get("headline", "")), 120),
                            evidence=normalize_text(str(it.get("evidence", "")), 600),
                            numbers=(it.get("numbers") if isinstance(it.get("numbers"), list) else []),
                            source_refs=(it.get("source_refs") if isinstance(it.get("source_refs"), list) else []),
                        )
                    )

            if not counters:
                # AI가 이상하게 주면 fallback 카운터라도 넣기
                counters = fallback_deep(payload, sources).counters

            questions = [normalize_text(q, 140) for q in (questions_raw[:3] if isinstance(questions_raw, list) else [])]
            if len(questions) < 3:
                questions = fallback_deep(payload, sources).questions

            resp = DeepReportResponse(
                asOf=now_iso(),
                cached=False,
                asset=payload.asset,
                ticker=payload.ticker,
                company=payload.company,
                conviction=payload.conviction,
                summary=summary,
                counters=counters,
                questions=questions,
                sources=sources,
            )

            out = resp.model_dump()
            cache.set(ck, out)
            return out

        resp = fallback_deep(payload, sources).model_dump()
        cache.set(ck, resp)
        return resp

    except Exception as e:
        logger.warning(f"deep-report ai failed -> fallback: {e}")
        resp = fallback_deep(payload, sources).model_dump()
        cache.set(ck, resp)
        return resp
