# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import copy
import itertools
import logging
import os
import uuid
import sys
import traceback
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response

# ==============================================================================
# 1. Logging Configuration
# ==============================================================================
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
        )
logger = logging.getLogger("vllm-proxy")

# Mute httpx default logging to avoid clutter
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==============================================================================
# 2. Resiliency Configuration
# ==============================================================================
MAX_RETRIES = 3
RETRY_DELAY_SEC = 1.0

# Prevent indefinite hangs.
# connect: 5s to fail-fast dead nodes.
# read: 300s generous allowance for LLM generation/JSON generation.
HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=15.0, pool=10.0)
HTTP_LIMITS = httpx.Limits(max_connections=2000, max_keepalive_connections=500)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown events."""
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}"
        app.state.prefill_clients.append(
                {
                    "client": httpx.AsyncClient(
                            timeout=HTTP_TIMEOUT,
                            limits=HTTP_LIMITS,
                            base_url=prefiller_base_url,
                            ),
                    "host":   host,
                    "port":   port,
                    "id":     f"prefill-{i}-{host}:{port}"
                    }
                )

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}"
        app.state.decode_clients.append(
                {
                    "client": httpx.AsyncClient(
                            timeout=HTTP_TIMEOUT,
                            limits=HTTP_LIMITS,
                            base_url=decoder_base_url,
                            ),
                    "host":   host,
                    "port":   port,
                    "id":     f"decode-{i}-{host}:{port}"
                    }
                )

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    logger.info(
        f"Initialized {len(app.state.prefill_clients)} prefill clients and {len(app.state.decode_clients)} decode clients."
        )
    yield

    # Shutdown: Close all HTTP connections gracefully
    logger.info("Shutting down... Closing connection pools.")
    for client_info in app.state.prefill_clients + app.state.decode_clients:
        await client_info["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Disaggregated Router Proxy")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")  # Better default for Docker deployments

    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8100])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8200])

    args = parser.parse_args()

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def get_next_client(app: FastAPI, service_type: str):
    """Get the next client in round-robin fashion."""
    if service_type == "prefill":
        return app.state.prefill_clients[next(app.state.prefill_iterator)]
    elif service_type == "decode":
        return app.state.decode_clients[next(app.state.decode_iterator)]
    raise ValueError(f"Unknown service type: {service_type}")


async def send_prefill_request_with_retries(
        app: FastAPI, endpoint: str, original_req_data: dict, request_id: str, headers: dict
        ):
    # Deep copy to avoid mutating the original request dict across retries
    payload = copy.deepcopy(original_req_data)

    payload["kv_transfer_params"] = {
        "do_remote_decode":  True,
        "do_remote_prefill": False,
        "remote_engine_id":  None,
        "remote_block_ids":  None,
        "remote_host":       None,
        "remote_port":       None,
        }
    payload["stream"] = False
    payload["max_tokens"] = 1

    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload.pop("stream_options", None)

    # These args are not supported for Prefill phase
    min_tokens = payload.pop("min_tokens", None)
    min_completion_tokens = payload.pop("min_completion_tokens", None)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        client_info = get_next_client(app, "prefill")
        target_id = client_info["id"]

        try:
            logger.info(f"[{request_id}] Prefill attempt {attempt}/{MAX_RETRIES} -> {target_id}")
            async with client_info["client"].post(endpoint, json=payload, headers=headers) as response:

                # If client sent a bad request (4xx), return it immediately without retry.
                if 400 <= response.status_code < 500:
                    error_content = await response.aread()
                    logger.warning(f"[{request_id}] Prefill 4xx Error on {target_id}: {response.status_code}")
                    return response.status_code, error_content, None, None, None

                response.raise_for_status()
                response_json = response.json()
                await response.aread()  # Consume to release connection back to pool safely

                logger.info(f"[{request_id}] Prefill Success -> {target_id}")
                return 200, response_json, min_tokens, min_completion_tokens, client_info

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            last_error = e
            logger.warning(f"[{request_id}] Prefill failed on {target_id}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_SEC)

    logger.error(f"[{request_id}] All prefill retries exhausted.")
    raise last_error


async def stream_decode_with_retries(app: FastAPI, endpoint: str, req_data: dict, request_id: str, headers: dict):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        client_info = get_next_client(app, "decode")
        target_id = client_info["id"]

        try:
            logger.info(f"[{request_id}] Decode connection attempt {attempt}/{MAX_RETRIES} -> {target_id}")
            # Build request manually so we can send with stream=True while catching network errors up-front
            req = client_info["client"].build_request("POST", endpoint, json=req_data, headers=headers)
            response = await client_info["client"].send(req, stream=True)

            if 400 <= response.status_code < 500:
                error_content = await response.aread()
                await response.aclose()
                logger.warning(f"[{request_id}] Decode 4xx Error on {target_id}: {response.status_code}")
                return response.status_code, error_content, None

            response.raise_for_status()
            logger.info(f"[{request_id}] Decode connection established -> {target_id}")
            return 200, None, response

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            last_error = e
            logger.warning(f"[{request_id}] Decode connection failed on {target_id}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_SEC)

    logger.error(f"[{request_id}] All decode retries exhausted.")
    raise last_error


async def _handle_completions(api: str, request: Request):
    # Unique ID for tracing logs
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4())[:12])

    try:
        req_data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload"})

    model = req_data.get("model", "unknown")
    logger.info(f"[{request_id}] -> New request to {api} for model '{model}'")

    # Pass client's auth header or fallback to environment variables
    auth_header = request.headers.get("Authorization", f"Bearer {os.environ.get('OPENAI_API_KEY', '')}")
    headers = {
        "Authorization": auth_header,
        "X-Request-Id":  request_id,
        }

    try:
        # ==========================================
        # STEP 1: PREFILL PHASE
        # ==========================================
        p_status, p_data, min_t, min_comp_t, _ = await send_prefill_request_with_retries(
                request.app, api, req_data, request_id, headers
                )

        # Fast exit if proxy received 4xx error (e.g. max tokens exceeded, JSON schema validation failed)
        if p_status != 200:
            return Response(status_code=p_status, content=p_data, media_type="application/json")

        # ==========================================
        # STEP 2: PREPARE DECODE PAYLOAD
        # ==========================================
        decode_req_data = copy.deepcopy(req_data)

        if kv_transfer_params := p_data.get("kv_transfer_params"):
            decode_req_data["kv_transfer_params"] = kv_transfer_params

        if min_t is not None:
            decode_req_data["min_tokens"] = min_t
        if min_comp_t is not None:
            decode_req_data["min_completion_tokens"] = min_comp_t

        # ==========================================
        # STEP 3: DECODE PHASE
        # ==========================================
        d_status, d_error, d_response = await stream_decode_with_retries(
                request.app, api, decode_req_data, request_id, headers
                )

        if d_status != 200:
            return Response(status_code=d_status, content=d_error, media_type="application/json")

        # Generator to stream the response safely
        async def generate_stream():
            try:
                async for chunk in d_response.aiter_bytes():
                    yield chunk
            except Exception as e:
                logger.error(f"[{request_id}] Stream interrupted: {str(e)}")
            finally:
                # CRITICAL: Always release socket connection even if user disconnects abruptly
                await d_response.aclose()
                logger.debug(f"[{request_id}] Stream context closed.")

        return StreamingResponse(
                generate_stream(),
                media_type=d_response.headers.get("content-type", "application/json"),
                status_code=d_response.status_code
                )

    except Exception as e:
        logger.error(f"[{request_id}] Unhandled internal proxy error:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": {"message": "Internal Proxy Server Error"}})


# ==============================================================================
# Endpoints
# ==============================================================================

@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions(request.url.path, request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions(request.url.path, request)


@app.get("/v1/models")
async def get_models(request: Request):
    """Proxy the models request to one of the backend vLLM instances."""
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4())[:12])
    auth_header = request.headers.get("Authorization", f"Bearer {os.environ.get('OPENAI_API_KEY', '')}")
    headers = {"Authorization": auth_header, "X-Request-Id": request_id}

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        # We can just fetch this from the prefill pool, as it will mirror the model loaded
        client_info = get_next_client(request.app, "prefill")
        target_id = client_info["id"]

        try:
            logger.info(f"[{request_id}] Fetching models attempt {attempt}/{MAX_RETRIES} -> {target_id}")
            # Fast timeout for metadata fetching
            response = await client_info["client"].get("/v1/models", headers=headers, timeout=5.0)

            response.raise_for_status()
            return JSONResponse(status_code=response.status_code, content=response.json())

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            last_error = e
            logger.warning(f"[{request_id}] Fetching models failed on {target_id}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_SEC)

    logger.error(f"[{request_id}] All retries exhausted for /v1/models.")
    return JSONResponse(status_code=500, content={"error": {"message": "Upstream servers unavailable"}})


@app.get("/health")
@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status":            "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances":  len(app.state.decode_clients),
        }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    # log_level warning prevents FastAPI's noisy default access logs from burying our custom logger
    uvicorn.run(app, host=global_args.host, port=global_args.port, log_level="warning")