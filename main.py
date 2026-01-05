import os
import json
import time
import hashlib
import logging
import asyncio
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field


# =========================
# Config
# =========================
APP_VERSION = os.getenv("APP_VERSION", "the-short-api-1")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "A1")  # 너가 선택한 A + 1
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24h
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "4096"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # Render에서 "gemini-3.0-flash"로 바꿔도 됨
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.4"))

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("the-short")


# =========================
# Optional Gemini SDK
# =========================
_HAS_GEMINI = False
genai = None
try:
    import google.generativeai as genai  # pip: google-generativeai
    _HAS_GEMINI = True
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.warning("Gemini SDK not available or init failed: %s", e)
    _HAS_GEMINI = False
    genai = None


# =========================
# Cache (in-memory TTL)
# =========================
class TTLMemoryCache:
    def _init_(self, ttl_seconds: int, max_items: int):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: Dict[str, Any] = {}   # key -> {"exp": float, "val": Any}
        self._lock = asyncio.Lock()

    async def get(self, key: str):
        async with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            if item["exp"] < time.time():
                self._store.pop(key, None)
                return None
            return item["val"]

    async def set(self, key: str, value: Any):
        async with self._lock:
            # crude eviction if too big
            if len(self._store) >= self.max_items:
                # remove expired first
                now = time.time()
                expired = [k for k, v in self._store.items() if v["exp"] < now]
                for k in expired[:512]:
                    self._store.pop(k, None)

                # if still too big, remove random-ish oldest by exp
                if len(self._store) >= self.max_items:
                    oldest = sorted(self._store.items(), key=lambda kv: kv[1]["exp"])[:256]
                    for k, _ in oldest:
                        self._store.pop(k, None)

            self._store[key] = {"exp": time.time() + self.ttl, "val": value}


cache = TTLMemoryCache(CACHE_TTL_SECONDS, CACHE_MAX_ITEMS)


# =========================
# Models
# =========================
class Asset(str, Enum):
    US = "US"
    KR = "KR"
    COIN = "COIN"


class Severity(str, Enum):
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


class SourceLink(BaseModel):
    title: str
    url: HttpUrl
    domain: Optional[str] = None


class Blindspot(BaseModel):
    title: str
    valueLine: str
    detail: str
    severity: Severity


class AnalyzePayload(BaseModel):
    asset: Asset
    ticker: str
    company: str
    conviction: str
    locale: str = "ko-KR"
    forceRefresh: bool = False

    # 안전장치(너무 긴 입력 방지)
    def normalized(self) -> "AnalyzePayload":
        t = self.ticker.strip().upper()
        c = (self.company or "").strip()
        conv = (self.conviction or "").strip().replace("\n", " ")
        if len(conv) > 400:
            conv = conv[:400]
        return AnalyzePayload(
            asset=self.asset,
            ticker=t,
            company=c if c else "Unknown",
            conviction=conv,
            locale=self.locale,
            forceRefresh=self.forceRefresh,
        )


class AnalyzeResponse(BaseModel):
    asOf: Optional[str] = None
    blindspot: Blindspot
    questions: List[str] = Field(default_factory=list)
    sources: List[SourceLink] = Field(default_factory=list)


class DeepReportPayload(BaseModel):
    asset: Asset
    ticker: str
    company: str
    conviction: str
    locale: str = "ko-KR"
    forceRefresh: bool = False

    def normalized(self) -> "DeepReportPayload":
        t = self.ticker.strip().upper()
        c = (self.company or "").strip()
        conv = (self.conviction or "").strip().replace("\n", " ")
        if len(conv) > 400:
            conv = conv[:400]
        return DeepReportPayload(
            asset=self.asset,
            ticker=t,
            company=c if c else "Unknown",
            conviction=conv,
            locale=self.locale,
            forceRefresh=self.forceRefresh,
        )


class CounterEvidence(BaseModel):
    title: str
    factLine: str
    detail: str
    source: SourceLink


class DeepReportResponse(BaseModel):
    asOf: Optional[str] = None
    headline: str
    counterEvidence: List[CounterEvidence] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    sources: List[SourceLink] = Field(default_factory=list)


# =========================
# Utility
# =========================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def url_domain(u: str) -> str:
    try:
        d = urlparse(u).netloc
        return d.replace("www.", "")
    except Exception:
        return ""


def canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_cache_key(endpoint: str, payload_dict: Dict[str, Any]) -> str:
    raw = f"{APP_VERSION}:{PROMPT_VERSION}:{GEMINI_MODEL}:{endpoint}:{canonical_json(payload_dict)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def unique_sources(sources: List[SourceLink]) -> List[SourceLink]:
    seen = set()
    out = []
    for s in sources:
        key = str(s.url)
        if key in seen:
            continue
        seen.add(key)
        if not s.domain:
            s.domain = url_domain(str(s.url))
        out.append(s)
    return out[:10]


def build_source_candidates(asset: Asset, ticker: str, company: str) -> List[SourceLink]:
    t = ticker.upper().strip()
    sources: List[SourceLink] = []

    if asset == Asset.US:
        # 무조건 실존하는 URL만
        sources += [
            SourceLink(title=f"Yahoo Finance — {t}", url=f"https://finance.yahoo.com/quote/{t}", domain="finance.yahoo.com"),
            SourceLink(title=f"Nasdaq — {t}", url=f"https://www.nasdaq.com/market-activity/stocks/{t.lower()}", domain="nasdaq.com"),
            SourceLink(title=f"SEC EDGAR Search — {t}", url=f"https://www.sec.gov/edgar/search/#/q={t}", domain="sec.gov"),
        ]
    elif asset == Asset.KR:
        # KR은 6자리 숫자일 때 특히 강함
        code = t
        sources += [
            SourceLink(title=f"Naver Finance — {code}", url=f"https://finance.naver.com/item/main.nhn?code={code}", domain="finance.naver.com"),
            SourceLink(title=f"FnGuide — {code}", url=f"https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A{code}&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701", domain="comp.fnguide.com"),
            SourceLink(title="KRX — Korea Exchange", url="https://www.krx.co.kr/main/main.jsp", domain="krx.co.kr"),
        ]
    else:
        # COIN: 확실히 존재하는 검색/홈 위주 (심볼->slug는 불확실하니까)
        q = t
        sources += [
            SourceLink(title=f"CoinMarketCap Search — {q}", url=f"https://coinmarketcap.com/search/?q={q}", domain="coinmarketcap.com"),
            SourceLink(title=f"CoinGecko Search — {q}", url=f"https://www.coingecko.com/en/search?query={q}", domain="coingecko.com"),
            SourceLink(title="Token Unlocks — Schedule", url="https://token.unlocks.app/", domain="token.unlocks.app"),
        ]

    return unique_sources(sources)


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Gemini가 JSON 앞뒤로 말 붙여도 최대한 JSON만 추출.
    """
    text = text.strip()
    # 1) 바로 JSON이면 파싱
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) 코드펜스 제거
    if "⁠  " in text:
        parts = [p.strip() for p in text.split("  ⁠") if p.strip()]
        # 보통 가운데가 json
        for p in parts:
            if p.startswith("{") and p.endswith("}"):
                try:
                    return json.loads(p)
                except Exception:
                    pass

    # 3) 첫 { 부터 마지막 } 까지
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start:end+1]
        return json.loads(chunk)

    raise ValueError("Model output is not valid JSON.")


# =========================
# Prompts (A1)
# =========================
ANALYZE_PROMPT = """
너는 '더쇼트(THE SHORT)'의 반박 리포트 엔진이다.

목표:
•⁠  ⁠사용자의 한 문장 매수/보유 논리를 '추천 없이' 검증한다.
•⁠  ⁠말투는 단정하고 차갑게. 그러나 비하/욕설/모욕은 금지.
•⁠  ⁠투자 조언(매수/매도 지시) 금지. 오직 "검증"과 "반증 질문"만.

출력은 반드시 JSON만. 설명/머리말/코드블럭 금지.
스키마:
{{
  "blindspot": {{
    "title": "...",
    "valueLine": "...",
    "detail": "...",
    "severity": "LOW|MED|HIGH"
  }},
  "questions": ["...", "...", "..."],
  "sources": [
    {{"title":"...","url":"https://...","domain":"..."}}
  ]
}}

규칙:
•⁠  ⁠sources는 제공된 후보 URL에서만 선택해라(새 URL 만들지 마라).
•⁠  ⁠구체 숫자를 '추정'으로 만들지 마라. 대신 "확인해야 할 숫자"를 제시해라.
"""

DEEP_PROMPT = """
너는 '더쇼트(THE SHORT)'의 심층 반박 리포트 엔진이다.

목표:
•⁠  ⁠사용자 논리(한 문장)를 축으로 반대 근거 3~5개를 만들어서 방어력을 시험한다.
•⁠  ⁠말투는 단정하고 차갑게. 그러나 비하/욕설/모욕은 금지.
•⁠  ⁠투자 조언(매수/매도 지시) 금지. 오직 검증/리스크/반증만.

출력은 반드시 JSON만. 설명/머리말/코드블럭 금지.
스키마:
{{
  "headline": "...",
  "counterEvidence": [
    {{
      "title": "...",
      "factLine": "...",
      "detail": "...",
      "source": {{"title":"...","url":"https://...","domain":"..."}}
    }}
  ],
  "questions": ["...", "...", "..."],
  "sources": [
    {{"title":"...","url":"https://...","domain":"..."}}
  ]
}}

규칙:
•⁠  ⁠counterEvidence는 3~5개.
•⁠  ⁠sources는 제공된 후보 URL에서만 선택해라(새 URL 만들지 마라).
•⁠  ⁠숫자를 만들어내지 마라. 대신 "확인해야 할 핵심 수치/지표"를 명확히 적어라.
"""


def build_analyze_user_message(p: AnalyzePayload, candidates: List[SourceLink]) -> str:
    return f"""
[입력]
asset: {p.asset.value}
ticker: {p.ticker}
company: {p.company}
conviction: {p.conviction}
locale: {p.locale}

[출처 후보 URL 목록]
{json.dumps([s.model_dump() for s in candidates], ensure_ascii=False, indent=2)}

요구:
•⁠  ⁠위 출처 후보 중 2~4개만 골라 sources로 넣어라.
•⁠  ⁠blindspot은 '한 방'으로.
•⁠  ⁠questions는 3개. 각 질문은 답을 회피하기 어렵게.
""".strip()


def build_deep_user_message(p: DeepReportPayload, candidates: List[SourceLink]) -> str:
    return f"""
[입력]
asset: {p.asset.value}
ticker: {p.ticker}
company: {p.company}
conviction: {p.conviction}
locale: {p.locale}

[출처 후보 URL 목록]
{json.dumps([s.model_dump() for s in candidates], ensure_ascii=False, indent=2)}

요구:
•⁠  ⁠counterEvidence 3~5개.
•⁠  ⁠각 evidence는 "이 논리가 깨지는 방식"이 달라야 한다. (수요/마진/경쟁/규제/밸류 중 최소 3개 커버)
•⁠  ⁠source는 후보에서만 선택.
•⁠  ⁠headline은 한 줄로: "지금 네 논리에서 가장 위험한 지점은 _" 톤으로.
""".strip()


# =========================
# Gemini call
# =========================
def _gemini_generate(prompt: str) -> str:
    if not (_HAS_GEMINI and genai and GEMINI_API_KEY):
        raise RuntimeError("Gemini not configured")

    model = genai.GenerativeModel(GEMINI_MODEL)

    # JSON 강제는 SDK 버전에 따라 동작이 다를 수 있어 "프롬프트로 강제" + 파싱 복구로 안전하게 간다.
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": GEMINI_TEMPERATURE,
            "max_output_tokens": 1200,
        },
    )
    text = getattr(resp, "text", None)
    if not text:
        # 일부 버전은 candidates[0].content.parts[0].text 형태
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            text = ""
    return text.strip()


async def gemini_json(system_prompt: str, user_message: str) -> Dict[str, Any]:
    # event-loop block 방지
    full_prompt = f"{system_prompt}\n\n{user_message}".strip()
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(None, _gemini_generate, full_prompt)
    return extract_json_object(text)


# =========================
# Fallback (Gemini 실패해도 200 리턴)
# =========================
def fallback_analyze(p: AnalyzePayload, candidates: List[SourceLink]) -> AnalyzeResponse:
    conv = p.conviction.lower()
    if "일론" in conv or "머스크" in conv:
        blind = Blindspot(
            title="인물 신뢰가 논리의 빈칸을 가린다",
            valueLine="‘그 사람이니까 된다’는 근거가 아니라 기대치입니다.",
            detail="실행 리스크(규제·경쟁·마진·일정 지연)가 터지면 논리는 즉시 무너집니다. 인물 기대를 수치/조건으로 바꿔야 검증이 됩니다.",
            severity=Severity.MED,
        )
    elif "저평가" in conv or "per" in conv:
        blind = Blindspot(
            title="저평가가 아니라, 정당한 할인일 수 있다",
            valueLine="싸서 오르는 게 아니라, 싫어할 이유가 사라져야 오릅니다.",
            detail="멀티플 회복은 자동이 아닙니다. 할인 요인이 유지되면 평균회귀는 오지 않습니다. '할인 해소 조건'을 명시해야 합니다.",
            severity=Severity.MED,
        )
    else:
        blind = Blindspot(
            title="핵심 전제가 문장에 없다",
            valueLine="‘왜 지금’과 ‘언제까지’가 비어있습니다.",
            detail="좋은 논리는 시간/조건을 포함합니다. 지금 문장은 신념에 가깝고, 검증 가능한 조건이 적습니다.",
            severity=Severity.MED,
        )

    qs = [
        "이 논리가 깨지는 조건 1개를 적을 수 있습니까?",
        "다음 분기에 반드시 확인할 ‘숫자/지표’ 1개는 무엇입니까?",
        "악재가 떠도 버틴다면, 언제까지/어떤 조건에서 철회합니까?",
    ]

    return AnalyzeResponse(
        asOf=now_iso(),
        blindspot=blind,
        questions=qs,
        sources=unique_sources(candidates[:3]),
    )


def fallback_deep(p: DeepReportPayload, candidates: List[SourceLink]) -> DeepReportResponse:
    headline = "지금 네 논리에서 가장 위험한 지점은 ‘검증 가능한 조건’이 비어있는 겁니다."
    sources = unique_sources(candidates[:3])

    ev = []
    for i in range(3):
        src = sources[i % len(sources)]
        ev.append(
            CounterEvidence(
                title=f"반대 근거 #{i+1}: 확인 지표가 없다",
                factLine="‘오를 것’이 아니라 ‘무엇이 확인되면 유지할지’를 적어야 합니다.",
                detail="이 논리는 데이터가 붙기 전에 끝납니다. 다음 분기/다음 달에 확인할 지표를 정하지 않으면, 결과는 감정이 결정합니다.",
                source=src,
            )
        )

    qs = [
        "이 논리로 ‘유지’할 조건 1개만 써보세요. (숫자/지표로)",
        "손절/축소 기준을 정할 수 있습니까? (가격이 아니라 조건)",
        "다음 업데이트에서 ‘반드시 확인’할 출처 1개를 고르세요.",
    ]

    return DeepReportResponse(
        asOf=now_iso(),
        headline=headline,
        counterEvidence=ev,
        questions=qs,
        sources=sources,
    )


# =========================
# FastAPI app
# =========================
app = FastAPI(title="THE SHORT API", version=APP_VERSION)

# iOS는 CORS 상관없지만, 나중에 웹 붙을 때 편해서 넣음
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    logger.exception("Unhandled error req_id=%s path=%s", req_id, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "request_id": req_id},
    )


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    return response


@app.get("/")
async def root():
    return {
        "ok": True,
        "service": "the-short",
        "version": APP_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": GEMINI_MODEL,
        "gemini_ready": bool(_HAS_GEMINI and GEMINI_API_KEY),
    }


@app.get("/health")
async def health():
    return {"ok": True, "ts": now_iso()}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzePayload, request: Request):
    p = payload.normalized()
    candidates = build_source_candidates(p.asset, p.ticker, p.company)

    key = make_cache_key("analyze", p.model_dump())
    if not p.forceRefresh:
        cached = await cache.get(key)
        if cached:
            return cached

    # Gemini 호출
    try:
        if _HAS_GEMINI and GEMINI_API_KEY:
            user_msg = build_analyze_user_message(p, candidates)
            data = await gemini_json(ANALYZE_PROMPT, user_msg)

            # 모델 출력 검증/정리
            blind = Blindspot(**data["blindspot"])
            qs = data.get("questions", [])[:3]
            srcs = [SourceLink(**s) for s in data.get("sources", [])]
            srcs = unique_sources(srcs) if srcs else unique_sources(candidates[:3])

            res = AnalyzeResponse(
                asOf=now_iso(),
                blindspot=blind,
                questions=qs if qs else fallback_analyze(p, candidates).questions,
                sources=srcs,
            )
        else:
            res = fallback_analyze(p, candidates)

    except Exception as e:
        # Gemini/파싱 실패해도 200으로 떨어뜨려서 앱이 안 죽게 함
        logger.warning("analyze fallback due to error: %s req_id=%s", e, request.state.request_id)
        res = fallback_analyze(p, candidates)

    await cache.set(key, res)
    return res


@app.post("/v1/deep-report", response_model=DeepReportResponse)
async def deep_report(payload: DeepReportPayload, request: Request):
    p = payload.normalized()
    candidates = build_source_candidates(p.asset, p.ticker, p.company)

    key = make_cache_key("deep-report", p.model_dump())
    if not p.forceRefresh:
        cached = await cache.get(key)
        if cached:
            return cached

    try:
        if _HAS_GEMINI and GEMINI_API_KEY:
            user_msg = build_deep_user_message(p, candidates)
            data = await gemini_json(DEEP_PROMPT, user_msg)

            headline = str(data.get("headline", "")).strip()
            raw_ev = data.get("counterEvidence", [])[:5]
            ev = [CounterEvidence(**x) for x in raw_ev]
            qs = data.get("questions", [])[:3]

            # sources 정리 (없으면 후보로 대체)
            srcs = [SourceLink(**s) for s in data.get("sources", [])]
            srcs = unique_sources(srcs) if srcs else unique_sources(candidates[:3])

            if not headline:
                headline = "지금 네 논리에서 가장 위험한 지점은 ‘검증 가능한 조건’이 비어있는 겁니다."
            if len(ev) < 3:
                # 부족하면 fallback 증거로 보강
                fb = fallback_deep(p, candidates)
                ev = (ev + fb.counterEvidence)[:3]

            res = DeepReportResponse(
                asOf=now_iso(),
                headline=headline,
                counterEvidence=ev[:5],
                questions=qs if qs else fallback_deep(p, candidates).questions,
                sources=srcs,
            )
        else:
            res = fallback_deep(p, candidates)

    except Exception as e:
        logger.warning("deep-report fallback due to error: %s req_id=%s", e, request.state.request_id)
        res = fallback_deep(p, candidates)

    await cache.set(key, res)
    return res
