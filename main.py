"""
SecondLook Backend (Render-ready) — FastAPI + Gemini 2.5 Flash + Symbol Master (US/KR)

변경 핵심 (2026-01)
•⁠  ⁠"기록/입력/질문 숙제" 제거
•⁠  ⁠/analyze 출력은 "취조 + 팩트 1문장 + 반대데이터 1개"로 고정
•⁠  ⁠팩트는:
  - US: Yahoo quoteSummary(서버에서 숫자 가져옴)
  - KR: pykrx(시총/PER/PBR/배당/20일 변동률 등)
•⁠  ⁠AI는 제공된 FACT 이외 숫자/주장 생성 금지 (환각 방지)

Endpoints
•⁠  ⁠/resolve : 입력 텍스트 -> 심볼 확정(resolved) 또는 후보(candidates)
•⁠  ⁠/analyze : 취조 결과 생성(팩트 기반)
•⁠  ⁠/symbols/* : 심볼 마스터 정적 제공
•⁠  ⁠/admin/symbols/update : (보호됨) 심볼 마스터 갱신

환경변수
•⁠  ⁠GEMINI_API_KEY
•⁠  ⁠GEMINI_MODEL (default: gemini-2.5-flash)
•⁠  ⁠ADMIN_TOKEN (optional)
•⁠  ⁠SYMBOL_DIR (default: ./data/symbols)
•⁠  ⁠CACHE_TTL_SECONDS (default: 86400)
•⁠  ⁠RESULT_CACHE_MAXSIZE (default: 4000)
•⁠  ⁠ENABLE_KR (default: 1)
"""

from _future_ import annotations

import asyncio
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from cachetools import TTLCache
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Optional: KR symbols + metrics via pykrx
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


GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")
SYMBOL_DIR = Path(_env("SYMBOL_DIR", "./data/symbols") or "./data/symbols")
ADMIN_TOKEN = _env("ADMIN_TOKEN", None)
ENABLE_KR = (_env("ENABLE_KR", "1") or "1") == "1"

CACHE_TTL_SECONDS = int(_env("CACHE_TTL_SECONDS", "86400") or "86400")
RESULT_CACHE_MAXSIZE = int(_env("RESULT_CACHE_MAXSIZE", "4000") or "4000")

DISCLAIMER_LINE = "SecondLook는 추천하지 않습니다. 결론은 사용자 판단입니다."

# 투자 조언으로 읽힐 표현 최소화 (앱/서버 공통 안전장치)
BANNED_WORDS = [
    "매수", "매도", "추천", "목표가", "타이밍", "확실", "수익",
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


class CounterChartHint(BaseModel):
    title: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None


class CounterData(BaseModel):
    axis: Literal[
        "adoption", "competition", "regulation", "monetization",
        "cashflow", "dilution", "macro", "other"
    ] = "other"
    counter_one_liner: str
    why_it_matters: str
    what_to_check: str
    chart_hint: Optional[CounterChartHint] = None


class AnalyzeRequest(BaseModel):
    symbol_id: Optional[str] = None
    q: Optional[str] = None
    locale: Literal["ko", "en"] = "ko"
    tier: Literal["free", "premium"] = "free"
    client_version: Optional[str] = None


class AnalyzeResponse(BaseModel):
    ticker: str
    display_name: str
    exchange: str
    country: str
    updated_at: str
    disclaimer: str

    interrogation_headline: str
    thesis_one_liner: str
    blindspot_fact: str
    counter_data: CounterData
    cross_exam: str
    context_paragraph: str

    fact_metrics: List[FactMetric]
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
    blob: str


def _norm(s: str) -> str:
    s = s.strip().upper()
    s = re.sub(r"[^0-9A-Z가-힣\.\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_aliases(ticker: str, name: str, name_local: str | None, extra: List[str] | None = None) -> List[str]:
    base = {ticker, name}
    if name_local:
        base.add(name_local)
    if extra:
        base.update(extra)

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
    def _init_(self, symbol_dir: Path) -> None:
        self.symbol_dir = symbol_dir
        self.data_version: str | None = None
        self._items: List[SymbolItem] = []
        self._by_id: Dict[str, SymbolItem] = {}
        self._lock = threading.RLock()

    def load_from_disk(self) -> bool:
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
        qn = _norm(query)
        top = candidates[0]
        top_score = _score_item(qn, top)
        second_score = _score_item(qn, candidates[1]) if len(candidates) > 1 else -1
        if top_score >= 95 and (top_score - second_score >= 10):
            return top, candidates[:limit]
        return None, candidates[:limit]

    def rebuild_and_save(self) -> Dict[str, Any]:
        self.symbol_dir.mkdir(parents=True, exist_ok=True)
        today = dt.datetime.utcnow().date().isoformat()

        items_us = build_us_symbols(today)
        items_kr: List[SymbolItem] = []
        if ENABLE_KR:
            try:
                items_kr = build_kr_symbols(today)
            except Exception:
                items_kr = []

        us_path = self.symbol_dir / "us.json.gz"
        kr_path = self.symbol_dir / "kr.json.gz"

        _write_market_gz(us_path, market="US", data_version=today, items=items_us)
        if items_kr:
            _write_market_gz(kr_path, market="KR", data_version=today, items=items_kr)

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
            merged: List[SymbolItem] = []
            merged.extend(items_us)
            merged.extend(items_kr)
            self._set_items(merged)

        return {"data_version": today, "us_count": len(items_us), "kr_count": len(items_kr)}

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
    qt = q.replace(" ", "")
    t = _norm(it.ticker)
    if qt == t.replace(" ", ""):
        return 100 + min(it.rank // 100, 10)
    if t.replace(" ", "").startswith(qt):
        return 95 + min(it.rank // 150, 6)

    for a in it.aliases:
        an = _norm(a).replace(" ", "")
        if qt == an:
            return 92 + min(it.rank // 200, 5)
    for a in it.aliases:
        an = _norm(a).replace(" ", "")
        if an.startswith(qt):
            return 88 + min(it.rank // 250, 4)

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

    dedup: Dict[str, SymbolItem] = {it.id: it for it in items}
    return list(dedup.values())


def _parse_nasdaqlisted(txt: str, today: str) -> List[SymbolItem]:
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("|")
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

        test_issue = row.get("Test Issue", "").strip().upper() == "Y"
        if test_issue:
            continue

        etf_flag = (row.get("ETF", "").strip().upper() == "Y")
        stype: SymbolType = "ETF" if etf_flag else "EQUITY"

        exchange = "NASDAQ"
        country = "US"
        sid = f"US:{exchange}:{sym}"
        aliases = _safe_aliases(sym, name, None, extra=[sym.replace(".", ""), name.replace(",", "")])
        rank = 500
        blob = _norm(" ".join([sym, name, " ".join(aliases)]))

        out.append(SymbolItem(
            id=sid, ticker=sym, name=name, name_local=None,
            exchange=exchange, country=country, type=stype, status="ACTIVE",
            aliases=aliases, rank=rank, updated_at=today, blob=blob
        ))
    return out


def _parse_otherlisted(txt: str, today: str) -> List[SymbolItem]:
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("|")
    out: List[SymbolItem] = []

    exch_map = {"A": "NYSEAMER", "N": "NYSE", "P": "NYSEARCA", "Z": "BATS", "V": "IEX"}

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
            id=sid, ticker=sym, name=name, name_local=None,
            exchange=exchange, country=country, type=stype, status="ACTIVE",
            aliases=aliases, rank=rank, updated_at=today, blob=blob
        ))
    return out


def build_kr_symbols(today: str) -> List[SymbolItem]:
    if krx_stock is None:
        raise RuntimeError("pykrx import failed")

    markets = ["KOSPI", "KOSDAQ", "KONEX"]
    out: List[SymbolItem] = []

    for m in markets:
        tickers = krx_stock.get_market_ticker_list(market=m)
        for t in tickers:
            name = krx_stock.get_market_ticker_name(t)
            sid = f"KR:{m}:{t}"
            aliases = _safe_aliases(t, name, name, extra=[t, name])
            rank = 600 if m == "KOSPI" else 450 if m == "KOSDAQ" else 200
            blob = _norm(" ".join([t, name, " ".join(aliases)]))
            out.append(SymbolItem(
                id=sid, ticker=t, name=name, name_local=name,
                exchange=m, country="KR", type="EQUITY", status="ACTIVE",
                aliases=aliases, rank=rank, updated_at=today, blob=blob
            ))

    dedup: Dict[str, SymbolItem] = {it.id: it for it in out}
    return list(dedup.values())


# =========================================================
# Facts (US: Yahoo / KR: pykrx)
# =========================================================

YAHOO_QUOTE_SUMMARY = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"


def _yahoo_fmt(v: Any) -> str | None:
    if isinstance(v, dict):
        if "fmt" in v and v["fmt"]:
            return str(v["fmt"])
        if "raw" in v and v["raw"] is not None:
            return str(v["raw"])
    return None


def _yahoo_get_quote_summary(ticker: str) -> Dict[str, Any] | None:
    params = {"modules": "price,financialData,defaultKeyStatistics,summaryDetail"}
    headers = {
        "user-agent": "Mozilla/5.0 (SecondLook; +https://secondlook.onrender.com)"
    }
    try:
        r = httpx.get(YAHOO_QUOTE_SUMMARY.format(ticker=ticker), params=params, headers=headers, timeout=12)
        if r.status_code != 200:
            return None
        payload = r.json()
        res = payload.get("quoteSummary", {}).get("result", None)
        if not res:
            return None
        return res[0]
    except Exception:
        return None


def _kr_latest_trading_day() -> str:
    # 최근 10일 안에서 거래일 찾기
    for i in range(0, 10):
        d = (dt.date.today() - dt.timedelta(days=i)).strftime("%Y%m%d")
        try:
            # 아무 시장이나 호출해서 비어있지 않으면 거래일로 간주
            df = krx_stock.get_market_ohlcv_by_ticker(d, market="KOSPI")  # type: ignore
            if df is not None and len(df.index) > 0:
                return d
        except Exception:
            continue
    return dt.date.today().strftime("%Y%m%d")


def _fmt_krw_marketcap(v: Any) -> str:
    try:
        n = int(v)
    except Exception:
        return "—"

    # 원 단위일 가능성이 높음(pykrx는 '시가총액' 원 단위 제공)
    # 보기 좋게 조/억으로 축약
    if n >= 10**12:
        return f"{n/10**12:.1f}조원"
    if n >= 10**8:
        return f"{n/10**8:.0f}억원"
    return f"{n:,}원"


def fetch_fact_metrics(symbol: SymbolItem) -> List[FactMetric]:
    out: List[FactMetric] = []

    if symbol.country == "US":
        qs = _yahoo_get_quote_summary(symbol.ticker)
        if not qs:
            return out

        price = qs.get("price", {})
        fin = qs.get("financialData", {})
        # price
        mcap = _yahoo_fmt(price.get("marketCap"))
        if mcap:
            out.append(FactMetric(label="Market Cap", value=mcap))

        # revenue / margin / fcf
        rev = _yahoo_fmt(fin.get("totalRevenue"))
        if rev:
            out.append(FactMetric(label="Revenue", value=rev))

        opm = _yahoo_fmt(fin.get("operatingMargins"))
        if opm:
            # yahoo fmt is already like "7.82%"
            out.append(FactMetric(label="Operating Margin", value=opm))

        gpm = _yahoo_fmt(fin.get("grossMargins"))
        if gpm:
            out.append(FactMetric(label="Gross Margin", value=gpm))

        fcf = _yahoo_fmt(fin.get("freeCashflow"))
        if fcf:
            out.append(FactMetric(label="Free Cash Flow", value=fcf))

        return out

    if symbol.country == "KR" and krx_stock is not None:
        day = _kr_latest_trading_day()
        market = symbol.exchange  # KOSPI/KOSDAQ/KONEX
        try:
            fdf = krx_stock.get_market_fundamental_by_ticker(day, market=market)  # type: ignore
            cdf = krx_stock.get_market_cap_by_ticker(day, market=market)  # type: ignore

            if symbol.ticker in fdf.index:
                row = fdf.loc[symbol.ticker]
                per = row.get("PER", None)
                pbr = row.get("PBR", None)
                div = row.get("DIV", None)

                if per is not None and str(per) != "nan":
                    out.append(FactMetric(label="PER", value=str(per)))
                if pbr is not None and str(pbr) != "nan":
                    out.append(FactMetric(label="PBR", value=str(pbr)))
                if div is not None and str(div) != "nan":
                    out.append(FactMetric(label="DIV", value=f"{div}%"))

            if symbol.ticker in cdf.index:
                cap = cdf.loc[symbol.ticker].get("시가총액", None)
                if cap is not None:
                    out.append(FactMetric(label="Market Cap", value=_fmt_krw_marketcap(cap)))

            # 20일 변동률
            try:
                end = dt.datetime.strptime(day, "%Y%m%d").date()
                start = end - dt.timedelta(days=35)
                ohlcv = krx_stock.get_market_ohlcv_by_date(start.strftime("%Y%m%d"), day, symbol.ticker)  # type: ignore
                if ohlcv is not None and len(ohlcv.index) >= 2:
                    first = float(ohlcv.iloc[0]["종가"])
                    last = float(ohlcv.iloc[-1]["종가"])
                    if first > 0:
                        chg = (last / first - 1.0) * 100.0
                        out.append(FactMetric(label="20D Change", value=f"{chg:+.1f}%"))
            except Exception:
                pass

        except Exception:
            return out

        return out

    return out


# =========================================================
# AI generation (팩트 기반 취조 출력)
# =========================================================

class AIGenOutput(BaseModel):
    interrogation_headline: str
    thesis_one_liner: str
    blindspot_fact: str
    counter_data: CounterData
    cross_exam: str
    context_paragraph: str


def _gemini_client():
    if genai is None:
        return None
    try:
        return genai.Client()
    except Exception:
        return None


GEMINI_CLIENT = _gemini_client()


def _extract_json(text: str) -> Dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        pass
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


def _clamp(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "…"


def _facts_to_lines(facts: List[FactMetric]) -> str:
    if not facts:
        return "- (FACT 없음)"
    return "\n".join([f"- {m.label}: {m.value}" for m in facts])


async def generate_bundle(symbol: SymbolItem, facts: List[FactMetric], locale: str) -> AIGenOutput:
    # Fallback if no Gemini
    if GEMINI_CLIENT is None:
        # 팩트가 없을 때도 "숫자 만들지 말기"
        any_fact = facts[0].value if facts else "—"
        return AIGenOutput(
            interrogation_headline=_clamp(_scrub_banned("좋아 보이는 이유는 누구나 말합니다. 지금은 ‘깨지는 이유’부터 봅니다."), 80),
            thesis_one_liner=_clamp(_scrub_banned("당신은 보통 ‘서사/기대’에 기대고 들어옵니다. 문제는 그 기대가 어디서 깨지는지입니다."), 120),
            blindspot_fact=_clamp(_scrub_banned(f"팩트 하나만 보죠: 현재 확인 가능한 수치는 {any_fact}입니다. 숫자가 흔들리면 서사가 먼저 무너집니다."), 160),
            counter_data=CounterData(
                axis="cashflow",
                counter_one_liner=_clamp(_scrub_banned("반대 데이터: ‘좋아질 것’이 아니라 ‘증명’이 늦어지면 논리가 먼저 붕괴합니다."), 140),
                why_it_matters=_clamp(_scrub_banned("서사는 버틸 수 있어도, 증명 지표는 먼저 가격에 반영됩니다."), 90),
                what_to_check=_clamp(_scrub_banned("마진/현금흐름 추세가 ‘연속’으로 꺾이는지 확인하세요."), 90),
                chart_hint=CounterChartHint(title="Margin / FCF trend", x="Quarter", y="% or currency"),
            ),
            cross_exam=_clamp(_scrub_banned("근거가 뭔데?"), 15),
            context_paragraph=_clamp(_scrub_banned("당신은 이 종목을 ‘좋은 이야기’로 보고 있을 가능성이 큽니다. 하지만 시장은 보통 숫자(마진/현금흐름)로 먼저 태도를 바꿉니다. 그래서 지금은 이야기보다 ‘증명’의 순서를 확인해야 합니다."), 240),
        )

    facts_lines = _facts_to_lines(facts)

    system = f"""
너는 SecondLook다. 역할은 "투자 조언"이 아니라 "확신을 깨는 점검"이다.
금지: 매수/매도/추천/목표가/타이밍/확실/수익, 그리고 buy/sell/recommend/target price/profit 등.
톤:
•⁠  ⁠선생님/리포트 톤 금지. 짧고 단정적으로.
•⁠  ⁠욕/비하/조롱 금지. 대신 도발적이어도 됨.
중요 규칙:
•⁠  ⁠FACT에 없는 숫자/수치/구체 주장(예: 15% 상승) 절대 생성하지 마라.
•⁠  ⁠blindspot_fact는 가능하면 FACT에 있는 숫자(%, B, 조원 등)를 최소 1개 포함해 1문장으로 작성하라.
•⁠  ⁠counter_data도 FACT 기반으로 작성하라. FACT가 부족하면 "확인해야 할 지표"로 말하되 숫자를 만들지 마라.
출력은 오직 JSON.
키:
•⁠  ⁠interrogation_headline (짧게, 1~2문장)
•⁠  ⁠thesis_one_liner (1문장: 사용자가 보통 이 종목에 기대는 전제)
•⁠  ⁠blindspot_fact (1문장: 팩트 한 방)
•⁠  ⁠counter_data (axis, counter_one_liner, why_it_matters, what_to_check, chart_hint)
•⁠  ⁠cross_exam (15자 이내: 한 줄 취조)
•⁠  ⁠context_paragraph (2~3문장, '당신은'으로 시작)
"""

    user = f"""
SYMBOL
•⁠  ⁠ticker: {symbol.ticker}
•⁠  ⁠name: {symbol.name_local or symbol.name}
•⁠  ⁠exchange: {symbol.exchange}
•⁠  ⁠country: {symbol.country}

FACT (여기 숫자만 사용 가능)
{facts_lines}

요구사항
•⁠  ⁠모든 문장은 한국어.
•⁠  ⁠blindspot_fact는 1문장.
•⁠  ⁠cross_exam은 15자 이내.
•⁠  ⁠counter_data.axis는 adoption|competition|regulation|monetization|cashflow|dilution|macro|other 중 하나.
•⁠  ⁠chart_hint는 "어떤 차트를 보면 이게 확인되는지" 힌트를 짧게.
"""

    def _call() -> str:
        resp = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL,
            contents=[system, user],
        )
        return getattr(resp, "text", "") or ""

    text = await asyncio.get_event_loop().run_in_executor(None, _call)
    payload = _extract_json(text)
    if not payload:
        # fallback safe
        return await generate_bundle(symbol, facts, locale="ko")  # reuse fallback path

    try:
        out = AIGenOutput.model_validate(payload)
    except Exception:
        return await generate_bundle(symbol, facts, locale="ko")

    # scrub + clamp
    out.interrogation_headline = _clamp(_scrub_banned(out.interrogation_headline), 90)
    out.thesis_one_liner = _clamp(_scrub_banned(out.thesis_one_liner), 140)
    out.blindspot_fact = _clamp(_scrub_banned(out.blindspot_fact), 200)
    out.cross_exam = _clamp(_scrub_banned(out.cross_exam), 15)
    out.context_paragraph = _clamp(_scrub_banned(out.context_paragraph), 280)

    out.counter_data.counter_one_liner = _clamp(_scrub_banned(out.counter_data.counter_one_liner), 180)
    out.counter_data.why_it_matters = _clamp(_scrub_banned(out.counter_data.why_it_matters), 120)
    out.counter_data.what_to_check = _clamp(_scrub_banned(out.counter_data.what_to_check), 120)

    return out


# =========================================================
# FastAPI app
# =========================================================

app = FastAPI(title="SecondLook API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

symbol_store = SymbolStore(SYMBOL_DIR)
result_cache: TTLCache = TTLCache(maxsize=RESULT_CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)
cache_lock = threading.RLock()

SYMBOL_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/symbols", StaticFiles(directory=str(SYMBOL_DIR), html=False), name="symbols")


def _cache_key(symbol_id: str, locale: str, model: str) -> str:
    raw = f"{symbol_id}|{locale}|{model}|v2"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@app.on_event("startup")
async def _startup() -> None:
    loaded = symbol_store.load_from_disk()
    if not loaded:
        try:
            symbol_store.rebuild_and_save()
        except Exception:
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
    if mode == "suggest":
        candidates = symbol_store.search(q, limit=limit, country=country)
        return ResolveResponse(
            query=q,
            mode="candidates" if candidates else "not_found",
            resolved=None,
            candidates=[to_candidate(it) for it in candidates],
            data_version=symbol_store.data_version,
        )

    resolved_item, candidates = symbol_store.resolve(q, limit=limit, country=country)
    if resolved_item:
        return ResolveResponse(
            query=q,
            mode="resolved",
            resolved=to_candidate(resolved_item),
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
        resolved_item, _ = symbol_store.resolve(req.q, limit=10)
        symbol = resolved_item

    if symbol is None:
        raise HTTPException(status_code=404, detail="symbol_not_found")

    ck = _cache_key(symbol.id, req.locale, GEMINI_MODEL)
    with cache_lock:
        cached = result_cache.get(ck)
    if cached:
        return AnalyzeResponse(**cached)

    facts = fetch_fact_metrics(symbol)
    ai = await generate_bundle(symbol, facts, req.locale)

    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    resp = AnalyzeResponse(
        ticker=symbol.ticker,
        display_name=symbol.name_local or symbol.name,
        exchange=symbol.exchange,
        country=symbol.country,
        updated_at=now,
        disclaimer=DISCLAIMER_LINE,
        interrogation_headline=ai.interrogation_headline,
        thesis_one_liner=ai.thesis_one_liner,
        blindspot_fact=ai.blindspot_fact,
        counter_data=ai.counter_data,
        cross_exam=ai.cross_exam,
        context_paragraph=ai.context_paragraph,
        fact_metrics=facts,
        meta={
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
        raise HTTPException(status_code=500, detail=f"update_failed: {type(e)._name_}")


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


if _name_ == "_main_":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--update-symbols", action="store_true")
    args = p.parse_args()

    if args.update_symbols:
        store = SymbolStore(SYMBOL_DIR)
        out = store.rebuild_and_save()
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("Run with: uvicorn main:app --reload")
