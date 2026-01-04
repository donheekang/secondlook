"""
SecondLook Backend (Render-ready) — FastAPI + Gemini 2.5 Flash + Symbol Master (US/KR)

핵심
- /resolve : 사용자가 입력한 텍스트를 심볼로 "해석"(1개면 resolved, 여러개면 candidates)
- /analyze : FACT(숫자) + CONTEXT/BLINDSPOT/QUESTIONS 생성 (Gemini)
- /symbols/* : 심볼 마스터(index.json, us.json.gz, kr.json.gz) 정적 제공
- /admin/symbols/update : (보호됨) 일 1회 갱신 파이프라인 트리거 (Render Cron Job에서 호출)

환경변수
- GEMINI_API_KEY: Gemini Developer API Key
- GEMINI_MODEL: 기본 "gemini-2.5-flash"
- ADMIN_TOKEN: /admin/* 보호용 Bearer 토큰 (미설정 시 admin endpoint 비활성화)
- SYMBOL_DIR: 기본 "./data/symbols"
- CACHE_TTL_SECONDS: 기본 86400 (24h)
- RESULT_CACHE_MAXSIZE: 기본 4000
- ENABLE_KR: "1"이면 KR 심볼 수집 시도(기본 1)
- ALLOW_CLIENT_PREMIUM: "1"이면 client가 보내는 tier를 신뢰(MVP용, 기본 1)

실행 (Render start command 예시)
  uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import asyncio
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from cachetools import TTLCache
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Optional: KR symbols via pykrx
try:
    from pykrx import stock as krx_stock  # type: ignore
except Exception:  # pragma: no cover
    krx_stock = None

# Optional: Gemini
try:
    from google import genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None


# =========================================================
# Config
# =========================================================

def _env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")  # docs: gemini-2.5-flash
SYMBOL_DIR = Path(_env("SYMBOL_DIR", "./data/symbols") or "./data/symbols")
ADMIN_TOKEN = _env("ADMIN_TOKEN", None)
ENABLE_KR = (_env("ENABLE_KR", "1") or "1") == "1"

CACHE_TTL_SECONDS = int(_env("CACHE_TTL_SECONDS", "86400") or "86400")
RESULT_CACHE_MAXSIZE = int(_env("RESULT_CACHE_MAXSIZE", "4000") or "4000")
ALLOW_CLIENT_PREMIUM = (_env("ALLOW_CLIENT_PREMIUM", "1") or "1") == "1"

# One-line disclaimer must be returned always
DISCLAIMER_LINE = "SecondLook는 추천하지 않습니다. 놓치기 쉬운 관점만 짚어줍니다."

BANNED_WORDS = [
    # KR
    "매수", "매도", "추천", "목표가", "타이밍", "확실", "수익",
    # EN (최소)
    "buy", "sell", "recommend", "target price", "guarantee", "profit"
]

# =========================================================
# Models (API)
# =========================================================

SymbolType = Literal["EQUITY", "ETF", "ADR", "REIT", "ETN", "ETC"]


class SymbolCandidate(BaseModel):
    id: str
    ticker: str
    name: str
    name_local: Optional[str] = None
    exchange: str
    country: str
    type: SymbolType = "EQUITY"


class ResolveResponse(BaseModel):
    query: str
    mode: Literal["resolved", "candidates", "not_found"]
    resolved: Optional[SymbolCandidate] = None
    candidates: List[SymbolCandidate] = Field(default_factory=list)
    data_version: Optional[str] = None


class FactMetric(BaseModel):
    label: str
    value: str
    note: Optional[str] = None


class AnalyzeRequest(BaseModel):
    # Prefer symbol_id. If absent, server will resolve by q.
    symbol_id: Optional[str] = None
    q: Optional[str] = None

    locale: Literal["ko", "en"] = "ko"

    # free/premium gating (MVP: can trust client; later: receipt verification)
    tier: Literal["free", "premium"] = "free"

    # app versioning
    client_version: Optional[str] = None


class AnalyzeResponse(BaseModel):
    ticker: str
    display_name: str
    exchange: str
    country: str

    updated_at: str
    disclaimer: str

    fact_metrics: List[FactMetric]
    context_paragraph: str

    blindspot_paragraph: Optional[str] = None
    questions: List[str] = Field(default_factory=list)

    meta: Dict[str, Any] = Field(default_factory=dict)


# =========================================================
# Symbol Store
# =========================================================

@dataclass(frozen=True)
class SymbolItem:
    id: str
    ticker: str
    name: str
    name_local: str | None
    exchange: str
    country: str
    type: SymbolType
    status: str
    aliases: List[str]
    rank: int
    updated_at: str

    # Precomputed normalized blob for search
    blob: str


def _norm(s: str) -> str:
    s = s.strip().upper()
    # keep dot in class tickers like BRK.B
    s = re.sub(r"[^0-9A-Z가-힣\.\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_aliases(ticker: str, name: str, name_local: str | None, extra: List[str] | None = None) -> List[str]:
    base = {ticker, name}
    if name_local:
        base.add(name_local)
    if extra:
        base.update(extra)
    # Add normalized variants (strip spaces)
    out = set()
    for x in base:
        x = x.strip()
        if not x:
            continue
        out.add(x)
        out.add(_norm(x))
        out.add(_norm(x).replace(" ", ""))
    return sorted(out)


class SymbolStore:
    """
    Loads symbol master into memory.
    Also can rebuild from sources (US from nasdaqtrader, KR from pykrx) and write to SYMBOL_DIR.
    """

    def __init__(self, symbol_dir: Path) -> None:
        self.symbol_dir = symbol_dir
        self.data_version: str | None = None
        self._items: List[SymbolItem] = []
        self._by_id: Dict[str, SymbolItem] = {}
        self._lock = threading.RLock()

    # ---------- public ----------
    def load_from_disk(self) -> bool:
        """Return True if loaded; False if files missing."""
        index_path = self.symbol_dir / "index.json"
        us_path = self.symbol_dir / "us.json.gz"
        kr_path = self.symbol_dir / "kr.json.gz"

        if not index_path.exists() or not us_path.exists():
            return False

        with self._lock:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.data_version = index.get("data_version")

            items: List[SymbolItem] = []
            items.extend(self._load_market_gz(us_path))
            if kr_path.exists():
                items.extend(self._load_market_gz(kr_path))

            self._set_items(items)
            return True

    def get_by_id(self, symbol_id: str) -> SymbolItem | None:
        with self._lock:
            return self._by_id.get(symbol_id)

    def search(self, query: str, limit: int = 8, country: str | None = None) -> List[SymbolItem]:
        q = _norm(query)
        if not q:
            return []

        with self._lock:
            scored: List[Tuple[int, SymbolItem]] = []
            for it in self._items:
                if country and it.country != country:
                    continue
                s = _score_item(q, it)
                if s > 0:
                    scored.append((s, it))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [it for _, it in scored[:limit]]

    def resolve(self, query: str, limit: int = 8, country: str | None = None) -> Tuple[SymbolItem | None, List[SymbolItem]]:
        candidates = self.search(query, limit=limit, country=country)
        if not candidates:
            return None, []

        # resolution rule: strong top score + clear gap
        qn = _norm(query)
        top = candidates[0]
        top_score = _score_item(qn, top)
        second_score = _score_item(qn, candidates[1]) if len(candidates) > 1 else -1

        if top_score >= 95 and (top_score - second_score >= 10):
            return top, candidates[:limit]
        return None, candidates[:limit]

    def rebuild_and_save(self) -> Dict[str, Any]:
        """
        Download US/KR lists and rebuild symbol master.
        Writes gz files + index.json, and loads into memory.
        """
        self.symbol_dir.mkdir(parents=True, exist_ok=True)

        today = dt.datetime.utcnow().date().isoformat()
        items_us = build_us_symbols(today)
        items_kr: List[SymbolItem] = []
        if ENABLE_KR:
            try:
                items_kr = build_kr_symbols(today)
            except Exception:
                # If KR build fails, keep empty (MVP safe). In production, keep previous file instead.
                items_kr = []

        # Write market files
        us_path = self.symbol_dir / "us.json.gz"
        kr_path = self.symbol_dir / "kr.json.gz"

        _write_market_gz(us_path, market="US", data_version=today, items=items_us)
        if items_kr:
            _write_market_gz(kr_path, market="KR", data_version=today, items=items_kr)
        else:
            # do not delete existing KR file automatically (could be used by app)
            if not kr_path.exists():
                pass

        # index
        index = {
            "schema_version": 1,
            "data_version": today,
            "generated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "ttl_hours": 24,
            "files": [
                {"market": "US", "url": "/symbols/us.json.gz", "sha256": _sha256_file(us_path), "count": len(items_us)}
            ],
        }
        if items_kr:
            index["files"].append(
                {"market": "KR", "url": "/symbols/kr.json.gz", "sha256": _sha256_file(kr_path), "count": len(items_kr)}
            )

        (self.symbol_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

        with self._lock:
            self.data_version = today
            # load into memory
            merged: List[SymbolItem] = []
            merged.extend(items_us)
            merged.extend(items_kr)
            self._set_items(merged)

        return {"data_version": today, "us_count": len(items_us), "kr_count": len(items_kr)}

    # ---------- private ----------
    def _set_items(self, items: List[SymbolItem]) -> None:
        self._items = items
        self._by_id = {it.id: it for it in items}

    def _load_market_gz(self, path: Path) -> List[SymbolItem]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.loads(f.read())
        out: List[SymbolItem] = []
        for s in payload.get("symbols", []):
            out.append(SymbolItem(
                id=s["id"],
                ticker=s["ticker"],
                name=s["name"],
                name_local=s.get("name_local"),
                exchange=s["exchange"],
                country=s["country"],
                type=s.get("type", "EQUITY"),
                status=s.get("status", "ACTIVE"),
                aliases=s.get("aliases", []),
                rank=int(s.get("rank", 0)),
                updated_at=s.get("updated_at", payload.get("data_version", "")),
                blob=_norm(" ".join([s.get("ticker",""), s.get("name",""), s.get("name_local",""), " ".join(s.get("aliases", []))])),
            ))
        return out


def _score_item(q: str, it: SymbolItem) -> int:
    """
    Simple scoring:
    - exact ticker match: 100
    - ticker prefix: 95
    - alias exact: 92
    - alias prefix: 88
    - name contains: 75
    - blob contains: 60
    + small bonus from rank
    """
    qt = q.replace(" ", "")
    t = _norm(it.ticker)
    if qt == t.replace(" ", ""):
        return 100 + min(it.rank // 100, 10)

    if t.replace(" ", "").startswith(qt):
        return 95 + min(it.rank // 150, 6)

    # aliases
    for a in it.aliases:
        an = _norm(a).replace(" ", "")
        if qt == an:
            return 92 + min(it.rank // 200, 5)
    for a in it.aliases:
        an = _norm(a).replace(" ", "")
        if an.startswith(qt):
            return 88 + min(it.rank // 250, 4)

    # name / local
    nn = _norm(it.name)
    nl = _norm(it.name_local or "")
    if q in nn or q in nl:
        return 75 + min(it.rank // 300, 3)

    if q in it.blob:
        return 60

    return 0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_market_gz(path: Path, market: str, data_version: str, items: List[SymbolItem]) -> None:
    payload = {
        "market": market,
        "data_version": data_version,
        "symbols": [
            {
                "id": it.id,
                "ticker": it.ticker,
                "name": it.name,
                "name_local": it.name_local,
                "exchange": it.exchange,
                "country": it.country,
                "type": it.type,
                "status": it.status,
                "aliases": it.aliases,
                "rank": it.rank,
                "updated_at": it.updated_at,
            }
            for it in items
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(raw)


# =========================================================
# Symbol builders (US / KR)
# =========================================================

NASDAQ_NASDAQLISTED = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
NASDAQ_OTHERLISTED = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"


def build_us_symbols(today: str) -> List[SymbolItem]:
    # US list from Nasdaq Trader symbol directory files
    with httpx.Client(timeout=30) as client:
        r1 = client.get(NASDAQ_NASDAQLISTED)
        r1.raise_for_status()
        nasdaq_txt = r1.text

        r2 = client.get(NASDAQ_OTHERLISTED)
        r2.raise_for_status()
        other_txt = r2.text

    items: List[SymbolItem] = []
    items.extend(_parse_nasdaqlisted(nasdaq_txt, today))
    items.extend(_parse_otherlisted(other_txt, today))

    # de-dup by id
    dedup: Dict[str, SymbolItem] = {it.id: it for it in items}
    return list(dedup.values())


def _parse_nasdaqlisted(txt: str, today: str) -> List[SymbolItem]:
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return []

    header = lines[0].split("|")
    # expected first col: Symbol
    out: List[SymbolItem] = []
    for ln in lines[1:]:
        if ln.startswith("File Creation Time"):
            break
        cols = ln.split("|")
        if len(cols) < 2:
            continue
        row = dict(zip(header, cols))
        sym = row.get("Symbol", "").strip()
        name = row.get("Security Name", "").strip()
        if not sym or sym == "Symbol":
            continue

        # Filter test issues? MVP: keep ACTIVE only
        test_issue = row.get("Test Issue", "").strip().upper() == "Y"
        if test_issue:
            continue

        etf_flag = (row.get("ETF", "").strip().upper() == "Y")
        stype: SymbolType = "ETF" if etf_flag else "EQUITY"

        exchange = "NASDAQ"
        country = "US"
        sid = f"US:{exchange}:{sym}"

        aliases = _safe_aliases(sym, name, None, extra=[sym.replace(".", ""), name.replace(",", "")])
        rank = 500  # MVP baseline (you can replace with marketcap-based ranking later)
        blob = _norm(" ".join([sym, name, " ".join(aliases)]))
        out.append(SymbolItem(
            id=sid,
            ticker=sym,
            name=name,
            name_local=None,
            exchange=exchange,
            country=country,
            type=stype,
            status="ACTIVE",
            aliases=aliases,
            rank=rank,
            updated_at=today,
            blob=blob,
        ))
    return out


def _parse_otherlisted(txt: str, today: str) -> List[SymbolItem]:
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return []

    header = lines[0].split("|")
    out: List[SymbolItem] = []

    exch_map = {
        "A": "NYSEAMER",
        "N": "NYSE",
        "P": "NYSEARCA",
        "Z": "BATS",
        "V": "IEX",
    }

    for ln in lines[1:]:
        if ln.startswith("File Creation Time"):
            break
        cols = ln.split("|")
        if len(cols) < 3:
            continue
        row = dict(zip(header, cols))

        sym = row.get("ACT Symbol", "").strip()
        name = row.get("Security Name", "").strip()
        exch_code = row.get("Exchange", "").strip().upper()
        if not sym:
            continue

        exchange = exch_map.get(exch_code, f"EXCH_{exch_code}" if exch_code else "US")
        country = "US"
        sid = f"US:{exchange}:{sym}"

        etf_flag = (row.get("ETF", "").strip().upper() == "Y")
        stype: SymbolType = "ETF" if etf_flag else "EQUITY"

        aliases = _safe_aliases(sym, name, None, extra=[sym.replace(".", ""), name.replace(",", "")])
        rank = 350
        blob = _norm(" ".join([sym, name, " ".join(aliases)]))
        out.append(SymbolItem(
            id=sid,
            ticker=sym,
            name=name,
            name_local=None,
            exchange=exchange,
            country=country,
            type=stype,
            status="ACTIVE",
            aliases=aliases,
            rank=rank,
            updated_at=today,
            blob=blob,
        ))
    return out


def build_kr_symbols(today: str) -> List[SymbolItem]:
    """
    KR list:
    - MVP: pykrx 사용(스크래핑 방식이라 가끔 깨질 수 있음)
    - 실패하면 예외를 던짐(상위에서 처리)
    """
    if krx_stock is None:
        raise RuntimeError("pykrx is not installed or failed to import")

    markets = ["KOSPI", "KOSDAQ", "KONEX"]
    out: List[SymbolItem] = []

    for m in markets:
        tickers = krx_stock.get_market_ticker_list(market=m)
        for t in tickers:
            name = krx_stock.get_market_ticker_name(t)
            exchange = m
            country = "KR"
            sid = f"KR:{exchange}:{t}"
            aliases = _safe_aliases(t, name, name, extra=[t, name])
            # rank: prefer KOSPI slightly
            rank = 600 if m == "KOSPI" else 450 if m == "KOSDAQ" else 200
            blob = _norm(" ".join([t, name, " ".join(aliases)]))
            out.append(SymbolItem(
                id=sid,
                ticker=t,
                name=name,
                name_local=name,
                exchange=exchange,
                country=country,
                type="EQUITY",
                status="ACTIVE",
                aliases=aliases,
                rank=rank,
                updated_at=today,
                blob=blob,
            ))

    dedup: Dict[str, SymbolItem] = {it.id: it for it in out}
    return list(dedup.values())


# =========================================================
# AI generation (Gemini)
# =========================================================

class AIGenOutput(BaseModel):
    context_paragraph: str
    blindspot_paragraph: str
    questions: List[str]


def _gemini_client():
    # If GEMINI_API_KEY is set in env, google-genai Client() picks it up automatically
    if genai is None:
        return None
    try:
        return genai.Client()
    except Exception:
        return None


GEMINI_CLIENT = _gemini_client()


def _extract_json(text: str) -> Dict[str, Any] | None:
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Remove fenced blocks
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def _scrub_banned(text: str) -> str:
    out = text
    for w in BANNED_WORDS:
        out = re.sub(re.escape(w), "—", out, flags=re.IGNORECASE)
    return out


def _clamp(s: str, max_chars: int) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"


async def generate_context_blindspot_questions(
    *,
    symbol: SymbolItem,
    facts: List[FactMetric],
    locale: str,
) -> AIGenOutput:
    """
    Generate CONTEXT / BLINDSPOT / QUESTIONS as JSON.
    """
    # Fallback (if no Gemini)
    if GEMINI_CLIENT is None:
        return AIGenOutput(
            context_paragraph=_clamp(_scrub_banned(f"{symbol.ticker}는 지금 ‘기대’와 ‘확인’ 사이에서 평가가 갈리기 쉬운 구간입니다. 숫자 자체보다 맥락(무엇이 더 중요해졌는가)이 판단을 흔듭니다."), 280),
            blindspot_paragraph=_clamp(_scrub_banned("가장 흔한 착각은 ‘좋은 뉴스가 곧 회복’이라고 바로 연결하는 것입니다. 실제로는 수익 구조(마진/현금흐름)가 먼저 흔들릴 수 있습니다."), 360),
            questions=[
                _clamp(_scrub_banned("지금 내가 기대는 핵심 신호(숫자/가이던스)는 무엇인가?"), 120),
                _clamp(_scrub_banned("그 신호가 꺾이면, 나는 무엇을 바꿀 것인가?"), 120),
            ],
        )

    # Build prompt
    facts_lines = "\n".join([f"- {m.label}: {m.value}" + (f" ({m.note})" if m.note else "") for m in facts]) or "- (FACT 데이터 없음)"
    lang_hint = "Korean" if locale == "ko" else "English"

    system = f"""
You are SecondLook, an app that helps users double-check their thinking about a stock.
You MUST NOT give investment advice. Do not say buy/sell/recommend/target price/timing/profit/guarantee.
Write in {lang_hint}.
Return ONLY valid JSON with keys:
- context_paragraph: string (2-4 short lines)
- blindspot_paragraph: string (1 paragraph, strong)
- questions: array of 2-3 short strings

The blindspot must follow this template idea:
"가장 흔한 착각은 A다 → 실제로는 B가 먼저 흔들린다" (adapt naturally).
Keep it sharp and practical, but NOT prescriptive.
"""

    user = f"""
SYMBOL
- ticker: {symbol.ticker}
- name: {symbol.name}
- exchange: {symbol.exchange}
- country: {symbol.country}

FACT (numbers are provided; do not invent new numbers)
{facts_lines}

TASK
- Write a short CONTEXT paragraph that frames what the market is likely focusing on.
- Write one BLINDSPOT paragraph (the strongest line in the whole output).
- Provide 2-3 QUESTIONS to help the user check their own assumptions.
"""

    # Call Gemini (sync client inside threadpool for FastAPI)
    def _call() -> str:
        resp = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL,
            contents=[system, user],
        )
        # google-genai response supports .text in quickstart
        return getattr(resp, "text", "") or ""

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _call)

    payload = _extract_json(text)
    if not payload:
        # fallback: do not fail hard
        return AIGenOutput(
            context_paragraph=_clamp(_scrub_banned("맥락을 생성할 수 없어, 핵심 질문만 남깁니다."), 280),
            blindspot_paragraph=_clamp(_scrub_banned("가장 흔한 착각은 ‘정보가 부족하다’고 느끼는 것입니다. 실제로는 ‘내가 어떤 신호에 기대고 있는지’가 먼저 흔들릴 수 있습니다."), 360),
            questions=[
                _clamp(_scrub_banned("내가 기대는 핵심 신호는 무엇인가?"), 120),
                _clamp(_scrub_banned("그 신호가 무너지면 무엇을 바꿀 것인가?"), 120),
            ],
        )

    try:
        out = AIGenOutput.model_validate(payload)
    except Exception:
        return AIGenOutput(
            context_paragraph=_clamp(_scrub_banned("생성 결과를 해석하지 못했습니다. 다시 시도해 주세요."), 280),
            blindspot_paragraph=_clamp(_scrub_banned("가장 흔한 착각은 ‘내가 이미 충분히 이해했다’는 느낌입니다. 실제로는 변수의 우선순위가 바뀔 수 있습니다."), 360),
            questions=[_clamp(_scrub_banned("지금 내 판단의 근거는 무엇인가?"), 120)],
        )

    # Safety clamp & banned scrub
    out.context_paragraph = _clamp(_scrub_banned(out.context_paragraph), 280)
    out.blindspot_paragraph = _clamp(_scrub_banned(out.blindspot_paragraph), 360)
    out.questions = [_clamp(_scrub_banned(q), 120) for q in out.questions[:3]]

    return out


# =========================================================
# FACT provider (MVP stub)
# =========================================================

def fetch_fact_metrics(symbol: SymbolItem) -> List[FactMetric]:
    """
    MVP: 외부 재무 API 연동 전까지는 최소 신뢰용 '틀'만 유지.
    실제 운영에서는:
    - 분기/연간 매출 + YoY
    - 영업이익률(또는 GM)
    - FCF(가능 시)
    를 API로 채워 넣고, 여기서 반환.
    """
    # Basic heuristic placeholder
    if symbol.country == "KR" and re.fullmatch(r"\d{6}", symbol.ticker):
        return [
            FactMetric(label="매출", value="N/A", note="(데이터 연동 예정)"),
            FactMetric(label="영업이익률", value="N/A", note=None),
            FactMetric(label="FCF", value="N/A", note=None),
        ]
    return [
        FactMetric(label="Revenue", value="N/A", note="(wire up a finance API)"),
        FactMetric(label="Margin", value="N/A", note=None),
        FactMetric(label="FCF", value="N/A", note=None),
    ]


# =========================================================
# Entitlement / premium gating (No-login IAP friendly)
# =========================================================

def effective_tier(requested: str) -> Literal["free", "premium"]:
    """
    MVP: allow client-reported tier.
    Later: replace with server-side receipt verification / App Store Server API.
    """
    if not ALLOW_CLIENT_PREMIUM:
        return "free"
    return "premium" if requested == "premium" else "free"


def apply_gating(tier: Literal["free", "premium"], ai: AIGenOutput) -> Tuple[Optional[str], List[str]]:
    if tier == "premium":
        return ai.blindspot_paragraph, ai.questions

    # free: blindspot teaser + no questions
    teaser = ai.blindspot_paragraph.strip()
    teaser = _clamp(teaser, 90)
    return teaser, []


# =========================================================
# FastAPI app
# =========================================================

app = FastAPI(title="SecondLook API", version="0.1.0")

# CORS: tighten later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

symbol_store = SymbolStore(SYMBOL_DIR)

# Analysis cache (24h)
result_cache: TTLCache = TTLCache(maxsize=RESULT_CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)
cache_lock = threading.RLock()


def _cache_key(symbol_id: str, locale: str, tier: str, model: str) -> str:
    raw = f"{symbol_id}|{locale}|{tier}|{model}|v1"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Mount static symbols directory if exists
SYMBOL_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/symbols", StaticFiles(directory=str(SYMBOL_DIR), html=False), name="symbols")


@app.on_event("startup")
async def _startup() -> None:
    # try load; if missing, build once
    loaded = symbol_store.load_from_disk()
    if not loaded:
        # Build symbol master at first boot (can take a few seconds)
        try:
            symbol_store.rebuild_and_save()
        except Exception:
            # server can still run, but resolve may be limited
            pass


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "data_version": symbol_store.data_version, "model": GEMINI_MODEL}


@app.get("/resolve", response_model=ResolveResponse)
async def resolve(
    q: str = Query(..., min_length=1),
    country: str | None = Query(default=None, description="US 또는 KR로 필터"),
    limit: int = Query(8, ge=1, le=20),
    mode: Literal["resolve", "suggest"] = Query("resolve"),
) -> ResolveResponse:
    """
    mode=resolve: 1개로 확정되면 resolved로 반환
    mode=suggest: 항상 candidates만 반환 (자동완성용)
    """
    if mode == "suggest":
        candidates = symbol_store.search(q, limit=limit, country=country)
        return ResolveResponse(
            query=q,
            mode="candidates" if candidates else "not_found",
            resolved=None,
            candidates=[to_candidate(it) for it in candidates],
            data_version=symbol_store.data_version,
        )

    resolved, candidates = symbol_store.resolve(q, limit=limit, country=country)
    if resolved:
        return ResolveResponse(
            query=q,
            mode="resolved",
            resolved=to_candidate(resolved),
            candidates=[to_candidate(it) for it in candidates],
            data_version=symbol_store.data_version,
        )

    if candidates:
        return ResolveResponse(
            query=q,
            mode="candidates",
            resolved=None,
            candidates=[to_candidate(it) for it in candidates],
            data_version=symbol_store.data_version,
        )

    return ResolveResponse(query=q, mode="not_found", resolved=None, candidates=[], data_version=symbol_store.data_version)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    # resolve symbol
    symbol: SymbolItem | None = None
    if req.symbol_id:
        symbol = symbol_store.get_by_id(req.symbol_id)

    if symbol is None and req.q:
        resolved, _cands = symbol_store.resolve(req.q, limit=8)
        symbol = resolved

    if symbol is None:
        raise HTTPException(status_code=404, detail="symbol_not_found")

    tier = effective_tier(req.tier)

    # cache
    ck = _cache_key(symbol.id, req.locale, tier, GEMINI_MODEL)
    with cache_lock:
        cached = result_cache.get(ck)
    if cached:
        return AnalyzeResponse(**cached)

    facts = fetch_fact_metrics(symbol)
    ai = await generate_context_blindspot_questions(symbol=symbol, facts=facts, locale=req.locale)
    blindspot, questions = apply_gating(tier, ai)

    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    resp = AnalyzeResponse(
        ticker=symbol.ticker,
        display_name=symbol.name_local or symbol.name,
        exchange=symbol.exchange,
        country=symbol.country,
        updated_at=now,
        disclaimer=DISCLAIMER_LINE,
        fact_metrics=facts,
        context_paragraph=ai.context_paragraph,
        blindspot_paragraph=blindspot,
        questions=questions,
        meta={
            "tier": tier,
            "symbol_id": symbol.id,
            "data_version": symbol_store.data_version,
            "cached": False,
        },
    )

    with cache_lock:
        result_cache[ck] = resp.model_dump()

    return resp


# ---------- admin endpoints ----------

def require_admin(authorization: str | None = Header(default=None)) -> None:
    if ADMIN_TOKEN is None:
        raise HTTPException(status_code=404, detail="admin_disabled")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing_token")
    token = authorization.split(" ", 1)[1].strip()
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="invalid_token")


@app.post("/admin/symbols/update")
async def admin_update_symbols(_: Any = Depends(require_admin)) -> Dict[str, Any]:
    try:
        res = symbol_store.rebuild_and_save()
        return {"ok": True, **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"update_failed: {type(e).__name__}")


@app.post("/admin/cache/clear")
async def admin_cache_clear(_: Any = Depends(require_admin)) -> Dict[str, Any]:
    with cache_lock:
        result_cache.clear()
    return {"ok": True}


# =========================================================
# helpers
# =========================================================

def to_candidate(it: SymbolItem) -> SymbolCandidate:
    return SymbolCandidate(
        id=it.id,
        ticker=it.ticker,
        name=it.name,
        name_local=it.name_local,
        exchange=it.exchange,
        country=it.country,
        type=it.type,
    )


# =========================================================
# Optional CLI (local)
# =========================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--update-symbols", action="store_true", help="Build and write symbol master then exit")
    args = p.parse_args()

    if args.update_symbols:
        store = SymbolStore(SYMBOL_DIR)
        out = store.rebuild_and_save()
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("Run with: uvicorn main:app --reload")
