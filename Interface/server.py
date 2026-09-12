"""FastAPI backend for the single watch permission response."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Literal

from fastapi import FastAPI, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn


INTERFACE_DIR = Path(__file__).resolve().parent
PermissionCommand = Literal["H_ACCEPT", "H_REFUSE", "H_DEFER"]


class PermissionRequest(BaseModel):
    command: PermissionCommand


class PermissionEvent(BaseModel):
    event_type: PermissionCommand
    source: str
    human_step_id: int
    task_id: int
    timestamp: float


class PermissionAcknowledgement(BaseModel):
    accepted: bool
    event: PermissionEvent


app = FastAPI(
    title="HRC Watch Permission Demo",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def static_file(name: str) -> FileResponse:
    return FileResponse(INTERFACE_DIR / name, headers={"Cache-Control": "no-store"})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return static_file("index.html")


@app.get("/styles.css", include_in_schema=False)
def styles() -> FileResponse:
    return static_file("styles.css")


@app.get("/app.js", include_in_schema=False)
def javascript() -> FileResponse:
    return static_file("app.js")


@app.post(
    "/api/permission",
    response_model=PermissionAcknowledgement,
    status_code=status.HTTP_202_ACCEPTED,
)
def receive_permission(request: PermissionRequest) -> PermissionAcknowledgement:
    event = PermissionEvent(
        event_type=request.command,
        source="human_watch_demo",
        human_step_id=0,
        task_id=1,
        timestamp=time.time(),
    )
    print(f"Permission response received: {request.command}", flush=True)
    return PermissionAcknowledgement(accepted=True, event=event)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HRC watch permission demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
