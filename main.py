import os
from datetime import datetime, timezone
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AnyUrl, BaseModel, Field, conlist, constr

from google import genai

# -----------------------------
# Config
# -----------------------------
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
CLIENT_KEY = os.getenv("THE_SHORT_CLIENT_KEY", "").strip()  # optional: set to protect endpoint

# google-genai SDK reads GEMINI_API_KEY / GOOGLE_API_KEY from env if present
client = genai.Client()

app = FastAPI(title="The Short API", version="0.1.0")

# Native iOS app is not blocked by CORS, but enabling doesn't hurt (also helps testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class AnalyzeRequest(BaseModel):
    ticker: constr(strip_whitespace=True, min_length=1, max_length=12)
    company: constr(strip_whitespace=True, min_length=1, max_length=64) = "Unknown"
    conviction: constr(strip_whitespace=True, min_length=10, max_length=220)
    locale: constr(strip_whitespace=True, min_length=2, max_length=16) = "ko-KR"


class SourceOut(BaseModel):
    title: str = Field(..., max_length=140)
    url: str = Field(..., max_length=500)


class BlindspotOut(BaseModel):
    title: str = Field(..., max_length=90)
    value_line: str = Field(..., max_length=140)
    detail: str = Field(..., max_length=520)
    severity: Literal["low", "medium", "high"]


class LLMOut(BaseModel):
    blindspot: BlindspotOut
    questions: conlist(str, min_length=3, max_length=3)
    sources: List[SourceOut] = Field(default_factory=list)


class Source(BaseModel):
    title: str = Field(..., max_length=140)
    url: AnyUrl


class Blindspot(BaseModel):
    title: str
    value_line: str
    detail: str
    severity: Literal["low", "medium", "high"]


class AnalyzeResponse(BaseModel):
    blindspot: Blindspot
    questions: conlist(str, min_length=3, max_length=3)
    sources: List[Source] = Field(default_factory=list)
    as_of: str


# -----------------------------
# Auth (optional)
# -----------------------------
def require_client_key(x_client_key: Optional[str] = Header(None)) -> bool:
    if CLIENT_KEY:
        if not x_client_key or x_client_key != CLIENT_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")
    return True


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/analyze", response_model=AnalyzeResponse, dependencies=[Depends(require_client_key)])
def analyze(req: AnalyzeRequest):
    """
    Returns:
      - blindspot: single sharp counter-point
      - questions: 3 counter-audit questions
      - sources: optional grounding links (when tool available)
    """
    # "칼맛" 톤은 유지하되, 욕/모욕/매수·매도 지시 금지
    system = """
너는 '더쇼트(The Short)'의 분석 엔진이다.
목표는 사용자의 투자 논리를 공격적으로 '검증'하는 것이다.

규칙:
•⁠  ⁠투자 조언(매수/매도/보유 지시) 금지. "사라/팔아라" 같은 문장 금지.
•⁠  ⁠욕설/모욕/비하 표현 금지. 사용자를 공격하지 말고 논리를 공격해라.
•⁠  ⁠과장 금지. 확인 불가한 숫자/팩트는 만들지 마라.
•⁠  ⁠가능하면 공개 자료를 Google Search로 찾아서 근거(sources)를 남겨라.
•⁠  ⁠출력은 반드시 JSON 하나로만. 스키마에 맞춰라.

출력 형식:
•⁠  ⁠blindspot: (title, value_line, detail, severity)
•⁠  ⁠questions: 질문 3개 (각각 짧고 날카롭게)
•⁠  ⁠sources: 근거 링크 0~5개 (title, url)
"""

    user = f"""
[대상]
•⁠  ⁠ticker: {req.ticker}
•⁠  ⁠company: {req.company}

[사용자 논리(박제)]
{req.conviction}

[요청]
1) 사용자의 논리에서 가장 위험한 '빈 구멍' 하나만 고른다.
2) 그 빈 구멍을 한 줄 팩트(value_line)로 꽂는다. (검증 가능/과장 금지)
3) 확인 사살 질문 3개를 만든다. (손절 기준/검증 지표/반증 조건 같은 류)
4) 가능하면 sources에 URL을 남긴다.
"""

    prompt = system.strip() + "\n\n" + user.strip()

    # 1차: tools + structured outputs
    config_with_tools = {
        "tools": [{"google_search": {}}],
        "response_mime_type": "application/json",
        "response_json_schema": LLMOut.model_json_schema(),
        # Gemini 3는 temperature 기본값 유지 권장 (여기선 변경 안 함)
    }

    # 2차(폴백): tools 없이
    config_no_tools = {
        "response_mime_type": "application/json",
        "response_json_schema": LLMOut.model_json_schema(),
    }

    try:
        try:
            resp = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=config_with_tools,
            )
        except Exception:
            resp = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=config_no_tools,
            )

        if not getattr(resp, "text", None):
            raise HTTPException(status_code=502, detail="Empty model response")

        out = LLMOut.model_validate_json(resp.text)

        # sources 정리 (URL 형식만 기본 검증)
        sources: List[Source] = []
        for s in out.sources[:5]:
            try:
                sources.append(Source(title=s.title[:140], url=s.url))
            except Exception:
                # URL이 이상하면 버림 (클라이언트 디코딩 안정성)
                continue

        as_of = datetime.now(timezone.utc).isoformat()

        return AnalyzeResponse(
            blindspot=Blindspot(**out.blindspot.model_dump()),
            questions=out.questions,
            sources=sources,
            as_of=as_of,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model call failed: {type(e)._name_}")
