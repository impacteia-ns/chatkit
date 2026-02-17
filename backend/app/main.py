"""FastAPI entrypoint for exchanging workflow ids for ChatKit client secrets."""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import os
import uuid
from typing import Any, Mapping

import httpx
import requests
from fastapi import FastAPI, Request
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

DEFAULT_CHATKIT_BASE = "https://api.openai.com"
SESSION_COOKIE_NAME = "chatkit_session_id"
SESSION_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30  # 30 days

app = FastAPI(title="Managed ChatKit Session API")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def home():
    return FileResponse(str(STATIC_DIR / "index.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Mapping[str, str]:
    return {"status": "ok"}

from fastapi.responses import PlainTextResponse

@app.get("/.well-known/openai-apps-challenge")
async def openai_apps_challenge():
    return PlainTextResponse("Z3u9Lb9je2J_epZWgGAqUWcinsbzWW8gSZJXJdI5prM")


@app.post("/api/create-session")
async def create_session(request: Request) -> JSONResponse:
    """Exchange a workflow id for a ChatKit client secret."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return respond({"error": "Missing OPENAI_API_KEY environment variable"}, 500)

    body = await read_json_body(request)
    workflow_id = resolve_workflow_id(body)
    if not workflow_id:
        return respond({"error": "Missing workflow id"}, 400)

    user_id, cookie_value = resolve_user(request.cookies)
    api_base = chatkit_api_base()

    try:
        async with httpx.AsyncClient(base_url=api_base, timeout=10.0) as client:
            upstream = await client.post(
                "/v1/chatkit/sessions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "OpenAI-Beta": "chatkit_beta=v1",
                    "Content-Type": "application/json",
                },
                json={"workflow": {"id": workflow_id}, "user": user_id},
            )
    except httpx.RequestError as error:
        return respond(
            {"error": f"Failed to reach ChatKit API: {error}"},
            502,
            cookie_value,
        )
# ---------- CHATKIT PROXY (evita o frontend chamar a OpenAI direto) ----------

@app.post("/v1/chatkit/conversation")
async def proxy_chatkit_conversation(request: Request):
    body = await request.json()

    auth = request.headers.get("authorization")
    if not auth:
        return JSONResponse(status_code=400, content={"error": "Missing Authorization header"})

    async with httpx.AsyncClient(timeout=60) as client:
        upstream = await client.post(
            "https://api.openai.com/v1/chatkit/conversation",
            headers={
                "authorization": auth,
                "content-type": "application/json",
                "OpenAI-Beta": "chatkit-beta=v1",
            },
            json=body,
        )

    return JSONResponse(status_code=upstream.status_code, content=upstream.json())


@app.post("/v1/chatkit/domain_keys/verify_hosted")
async def proxy_verify_hosted(request: Request):
    body = await request.json()

    auth = request.headers.get("authorization")
    if not auth:
        return JSONResponse(status_code=400, content={"error": "Missing Authorization header"})

    async with httpx.AsyncClient(timeout=60) as client:
        upstream = await client.post(
            "https://api.openai.com/v1/chatkit/domain_keys/verify_hosted",
            headers={
                "authorization": auth,
                "content-type": "application/json",
                "OpenAI-Beta": "chatkit-beta=v1",
            },
            json=body,
        )

    return JSONResponse(status_code=upstream.status_code, content=upstream.json())

@app.post("/datacrazy/log")
async def datacrazy_log(payload: dict):
    datacrazy_token = os.getenv("DATACRAZY_API_TOKEN")

    if not datacrazy_token:
        return JSONResponse(
            status_code=500,
            content={"error": "DATACRAZY_API_TOKEN não configurado"}
        )

    return {
        "status": "ok",
        "received": payload
    }

    payload = parse_json(upstream)
    if not upstream.is_success:
        message = None
        if isinstance(payload, Mapping):
            message = payload.get("error")
        message = message or upstream.reason_phrase or "Failed to create session"
        return respond({"error": message}, upstream.status_code, cookie_value)

    client_secret = None
    expires_after = None
    if isinstance(payload, Mapping):
        client_secret = payload.get("client_secret")
        expires_after = payload.get("expires_after")

    if not client_secret:
        return respond(
            {"error": "Missing client secret in response"},
            502,
            cookie_value,
        )

    return respond(
        {"client_secret": client_secret, "expires_after": expires_after},
        200,
        cookie_value,
    )


def respond(
    payload: Mapping[str, Any], status_code: int, cookie_value: str | None = None
) -> JSONResponse:
    response = JSONResponse(payload, status_code=status_code)
    if cookie_value:
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=cookie_value,
            max_age=SESSION_COOKIE_MAX_AGE_SECONDS,
            httponly=True,
            samesite="lax",
            secure=is_prod(),
            path="/",
        )
    return response


def is_prod() -> bool:
    env = (os.getenv("ENVIRONMENT") or os.getenv("NODE_ENV") or "").lower()
    return env == "production"


async def read_json_body(request: Request) -> Mapping[str, Any]:
    raw = await request.body()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def resolve_workflow_id(body: Mapping[str, Any]) -> str | None:
    workflow = body.get("workflow", {})
    workflow_id = None
    if isinstance(workflow, Mapping):
        workflow_id = workflow.get("id")
    workflow_id = workflow_id or body.get("workflowId")
    env_workflow = os.getenv("CHATKIT_WORKFLOW_ID") or os.getenv(
        "VITE_CHATKIT_WORKFLOW_ID"
    )
    if not workflow_id and env_workflow:
        workflow_id = env_workflow
    if workflow_id and isinstance(workflow_id, str) and workflow_id.strip():
        return workflow_id.strip()
    return None


def resolve_user(cookies: Mapping[str, str]) -> tuple[str, str | None]:
    existing = cookies.get(SESSION_COOKIE_NAME)
    if existing:
        return existing, None
    user_id = str(uuid.uuid4())
    return user_id, user_id


def chatkit_api_base() -> str:
    return (
        os.getenv("CHATKIT_API_BASE")
        or os.getenv("VITE_CHATKIT_API_BASE")
        or DEFAULT_CHATKIT_BASE
    )


def parse_json(response: httpx.Response) -> Mapping[str, Any]:
    try:
        parsed = response.json()
        return parsed if isinstance(parsed, Mapping) else {}
    except (json.JSONDecodeError, httpx.DecodingError):
        return {}
   # fim do arquivo
# nenhuma indentação antes

@app.post("/datacrazy/log")
async def datacrazy_log(payload: dict):
    datacrazy_token = os.getenv("DATACRAZY_API_TOKEN")

    if not datacrazy_token:
        return JSONResponse(
            status_code=500,
            content={"error": "DATACRAZY_API_TOKEN não configurado"}
        )

    return {
        "status": "ok",
        "received": payload
    }
from fastapi.responses import PlainTextResponse

@app.get("/mcp")
async def mcp_root():
    return {
        "schema_version": "v1",
        "name": "Impacte IA MCP",
        "description": "MCP server mínimo para validação do ChatGPT Apps",
        "tools": []
    }
