# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from typing import Annotated

from fastapi import APIRouter, Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.tokencake.events import LifecycleEvent, StallFinished, StallStarted
from vllm.v1.engine.exceptions import EngineDeadError

logger = init_logger(__name__)
router = APIRouter()


@router.post("/v1/tokencake/events")
async def tokencake_event(
    event: Annotated[StallStarted | StallFinished, Body(discriminator="event")],
    raw_request: Request,
) -> JSONResponse:
    client: EngineClient | None = getattr(raw_request.app.state, "engine_client", None)
    settings = client.vllm_config._tokencake_config if client is not None else None
    if settings is None or not settings.offload.enabled or client is None:
        raise HTTPException(503, "TokenCake offload is unavailable")
    message = LifecycleEvent(
        event.event,
        event.lifecycle_id,
        event.kind if isinstance(event, StallStarted) else "stall",
        event.estimated_duration_s if isinstance(event, StallStarted) else None,
    )
    try:
        result = await asyncio.wait_for(client.tokencake_event(message), timeout=30.0)
    except (EngineDeadError, TimeoutError, NotImplementedError) as exc:
        raise HTTPException(503, "TokenCake offload is unavailable") from exc
    except Exception as exc:
        logger.exception("TokenCake event execution failed")
        raise HTTPException(500, "TokenCake event execution failed") from exc
    return JSONResponse(
        status_code=result.status_code,
        content={
            "lifecycle_id": result.lifecycle_id,
            "event": result.event,
            "state": result.state,
            "disposition": result.disposition,
        },
    )


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
