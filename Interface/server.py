"""FastAPI entry point for the H0 watch permission interface and HRC runtime."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn


INTERFACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INTERFACE_DIR.parent

# The original project uses layer directories rather than one installable Python
# package. Adding those directories here lets this standalone entry point import
# the exact Event, EventType, and runtime classes used by the rest of the system.
for layer in (
    "0_core",
    "1_recognition",
    "2_decision_making",
    "3_communication",
    "4_execution",
):
    path = str(REPO_ROOT / layer)
    if path not in sys.path:
        sys.path.insert(0, path)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from event_transport import UDPEventReceiver
from events import Event, EventType, RobotTaskState


# Literal gives FastAPI/Pydantic an explicit allowlist. A different command is
# rejected with HTTP 422 before the route function runs.
PermissionCommand = Literal["H_ACCEPT", "H_REFUSE", "H_DEFER"]


# Classes derived from BaseModel describe the JSON contract. FastAPI parses the
# request body into PermissionRequest and serializes the response models to JSON.
class PermissionRequest(BaseModel):
    command: PermissionCommand
    task_instance_id: str


class PermissionState(BaseModel):
    available: bool
    question: str | None = None
    task_instance_id: str | None = None
    human_step_id: int | None = None
    task_id: int | None = None


class PermissionEvent(BaseModel):
    event_type: PermissionCommand
    source: str
    task_instance_id: str
    human_step_id: int
    task_id: int
    timestamp: float


class PermissionAcknowledgement(BaseModel):
    processed: bool
    event: PermissionEvent


@dataclass(frozen=True)
class RuntimeSettings:
    """Values needed to run recognition input beside the HTTP server."""

    event_host: str = config.EVENT_TRANSPORT_HOST
    event_port: int = config.EVENT_TRANSPORT_PORT
    start_cli: bool = True
    debug_trigger: bool = False
    debug_round_id: int = 0
    debug_piece_id: int = 1


class HRCBridge:
    """Own one HRCSystem and connect HTTP, recognition, timers, voice, and ROS."""

    def __init__(
        self,
        system: Any,
        receiver: UDPEventReceiver | None = None,
        *,
        start_cli: bool = False,
        debug_trigger: bool = False,
        debug_round_id: int = 0,
        debug_piece_id: int = 1,
    ):
        self.system = system
        self.receiver = receiver
        self.start_cli = start_cli
        self.debug_trigger = debug_trigger
        self.debug_round_id = debug_round_id
        self.debug_piece_id = debug_piece_id
        self._process_lock = asyncio.Lock()
        self._process_task: asyncio.Task | None = None

    @classmethod
    def build_live(cls, settings: RuntimeSettings) -> "HRCBridge":
        """Build the same HRC runtime and recognition receiver as the CLI entry point."""

        # Importing here keeps module import side-effect free. The microphone,
        # ROS connection, UDP sender, and other runtime resources are created
        # only when FastAPI actually starts its application lifespan.
        from communication_runtime import build_system

        system = build_system()
        try:
            receiver = UDPEventReceiver(settings.event_host, settings.event_port)
        except Exception:
            system.close()
            raise
        return cls(
            system,
            receiver,
            start_cli=settings.start_cli,
            debug_trigger=settings.debug_trigger,
            debug_round_id=settings.debug_round_id,
            debug_piece_id=settings.debug_piece_id,
        )

    async def start(self) -> None:
        """Start the existing HRC event-processing loop inside FastAPI."""

        self.system.system_running = True
        if self.start_cli:
            self.system.start_cli_thread()
        if self.debug_trigger:
            self.system.event_queue.put(
                Event(
                    event_type=EventType.RECOGNITION_TRIGGER,
                    source="watch_debug_recognition",
                    payload={
                        "step_id": config.HUMAN_PULL_CABLES,
                        "piece_id": self.debug_piece_id,
                        "round_id": self.debug_round_id,
                        "progress": 1.0,
                    },
                )
            )
        self._process_task = asyncio.create_task(
            self._process_events(), name="hrc-event-loop"
        )

    async def stop(self) -> None:
        """Release HRC and recognition resources when FastAPI shuts down."""

        self.system.system_running = False
        if self._process_task is not None:
            self._process_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._process_task
            self._process_task = None
        if self.receiver is not None:
            self.receiver.close()
        self.system.close()

    async def _process_events(self) -> None:
        """Continuously feed recognition and queued events to TaskManager."""

        while self.system.system_running:
            if self.receiver is not None:
                for event in self.receiver.poll():
                    self.system.event_queue.put(event)
            async with self._process_lock:
                self.system.process_events()
            await asyncio.sleep(0.05)

    def _active_h0_permission(self):
        task = self.system.task_manager.active_task
        if task is None:
            return None
        if task.state is not RobotTaskState.R_WAITING_RESPONSE:
            return None
        if task.step_id != config.HUMAN_PULL_CABLES:
            return None
        if task.task_id != config.TASK_LIFT_PANEL:
            return None
        return task

    async def permission_state(self) -> PermissionState:
        """Return only the H0 permission that this UI is allowed to answer."""

        async with self._process_lock:
            task = self._active_h0_permission()
            if task is None:
                return PermissionState(available=False)
            return PermissionState(
                available=True,
                question="Lift the panel?",
                task_instance_id=task.task_instance_id,
                human_step_id=task.step_id,
                task_id=task.task_id,
            )

    async def submit_permission(self, payload: PermissionRequest) -> PermissionEvent:
        """Convert an HTTP command into the shared core Event and process it."""

        async with self._process_lock:
            task = self._active_h0_permission()
            if task is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="No Human Step 0 permission is waiting for a response.",
                )
            if payload.task_instance_id != task.task_instance_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="The permission belongs to an older task instance.",
                )

            # This is the key business-logic connection: the HTTP request becomes
            # the project's real Event and enters the same queue used by voice,
            # CLI, timers, recognition, and ROS callbacks.
            event = Event(
                event_type=EventType[payload.command],
                source="human_watch",
                task_instance_id=task.task_instance_id,
            )
            self.system.event_queue.put(event)
            self.system.process_events()

            return PermissionEvent(
                event_type=payload.command,
                source=event.source,
                task_instance_id=event.task_instance_id,
                human_step_id=task.step_id,
                task_id=task.task_id,
                timestamp=event.timestamp,
            )


def create_app(
    settings: RuntimeSettings | None = None,
    bridge: HRCBridge | None = None,
) -> FastAPI:
    """Create the FastAPI application, with an injectable bridge for tests."""

    runtime_settings = settings or RuntimeSettings()

    # A lifespan function is FastAPI's startup/shutdown context. The HRC runtime
    # is created once at startup, stored on app.state, and closed on shutdown.
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        active_bridge = bridge or HRCBridge.build_live(runtime_settings)
        application.state.hrc_bridge = active_bridge
        try:
            await active_bridge.start()
            yield
        finally:
            await active_bridge.stop()

    app = FastAPI(
        title="HRC Watch Permission",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # FastAPI calls dependencies before a route. This dependency retrieves the
    # single bridge stored during startup instead of constructing a new system
    # for every HTTP request.
    def get_bridge(request: Request) -> HRCBridge:
        return request.app.state.hrc_bridge

    def static_file(name: str) -> FileResponse:
        return FileResponse(
            INTERFACE_DIR / name,
            headers={"Cache-Control": "no-store"},
        )

    # Route decorators connect an HTTP method and path to a Python function.
    # FastAPI converts each returned FileResponse directly into an HTTP response.
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return static_file("index.html")

    @app.get("/styles.css", include_in_schema=False)
    def styles() -> FileResponse:
        return static_file("styles.css")

    @app.get("/app.js", include_in_schema=False)
    def javascript() -> FileResponse:
        return static_file("app.js")

    # This GET lets the watch discover the current real task instance. It keeps
    # a stale browser page from replying to a newer permission request.
    @app.get("/api/permission", response_model=PermissionState)
    async def current_permission(
        hrc_bridge: HRCBridge = Depends(get_bridge),
    ) -> PermissionState:
        return await hrc_bridge.permission_state()

    # FastAPI parses the JSON body as PermissionRequest before calling this route.
    # The response_model also verifies and documents the JSON returned to the UI.
    @app.post(
        "/api/permission",
        response_model=PermissionAcknowledgement,
        status_code=status.HTTP_200_OK,
    )
    async def receive_permission(
        payload: PermissionRequest,
        hrc_bridge: HRCBridge = Depends(get_bridge),
    ) -> PermissionAcknowledgement:
        event = await hrc_bridge.submit_permission(payload)
        return PermissionAcknowledgement(processed=True, event=event)

    return app


# ASGI servers can import this object as `server:app`. Running this file directly
# uses main() below so command-line options can configure the integrated runtime.
app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the watch interface with the HRC communication runtime."
    )
    parser.add_argument("--host", default="127.0.0.1", help="FastAPI bind host.")
    parser.add_argument("--port", default=8765, type=int, help="FastAPI port.")
    parser.add_argument("--event-host", default=config.EVENT_TRANSPORT_HOST)
    parser.add_argument("--event-port", default=config.EVENT_TRANSPORT_PORT, type=int)
    parser.add_argument(
        "--debug-trigger",
        action="store_true",
        help="Inject one Human Step 0 recognition trigger at startup.",
    )
    parser.add_argument("--debug-round-id", default=0, type=int)
    parser.add_argument("--debug-piece-id", default=1, type=int)
    args = parser.parse_args()

    settings = RuntimeSettings(
        event_host=args.event_host,
        event_port=args.event_port,
        debug_trigger=args.debug_trigger,
        debug_round_id=args.debug_round_id,
        debug_piece_id=args.debug_piece_id,
    )
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
