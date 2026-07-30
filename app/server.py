"""
ONSU Kitchen Assistant API

Run locally:

    uvicorn app.server:app --reload

"""

import os

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag import answer
from app.config import ONSU_PASSCODE


app = FastAPI(
    title="ONSU Kitchen Assistant"
)


# -----------------------------------------------------------------------------
# CORS
# -----------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request model
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):

    message: str


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok"
    }



@app.post("/chat")
def chat_endpoint(
    req: ChatRequest,
    x_api_key: str | None = Header(default=None),
):

    if ONSU_PASSCODE:

        if x_api_key != ONSU_PASSCODE:

            raise HTTPException(
                status_code=401,
                detail="Unauthorized"
            )


    if not req.message.strip():

        raise HTTPException(
            status_code=400,
            detail="Empty message"
        )


    response = answer(
        req.message
    )


    return {
        "answer": response
    }


# -----------------------------------------------------------------------------
# Frontend
# -----------------------------------------------------------------------------

app.mount(
    "/",
    StaticFiles(
        directory="static",
        html=True
    ),
    name="static"
)