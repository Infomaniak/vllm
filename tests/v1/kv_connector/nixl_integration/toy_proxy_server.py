# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import collections
import copy
import functools
import itertools
import logging
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# ==============================================================================
# 1. Logging Configuration
# ==============================================================================
logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", )
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


# ==============================================================================
# 3. Metrics Configuration & Tracker
# ==============================================================================
class ProxyMetrics:
    """Zero-dependency Prometheus metrics generator for basic proxy telemetry."""

    def __init__(self) -> None:
        self.requests_total: Dict[Tuple[str, str], int] = collections.defaultdict(int)
        self.prefill_total: Dict[str, int] = collections.defaultdict(int)
        self.decode_total: Dict[str, int] = collections.defaultdict(int)
        self.request_duration_sum: Dict[str, float] = collections.defaultdict(float)
        self.request_duration_count: Dict[str, int] = collections.defaultdict(int)

    def record_request(self, endpoint: str, status: int, duration_sec: float) -> None:
        self.requests_total[(endpoint, str(status))] += 1
        self.request_duration_sum[endpoint] += duration_sec
        self.request_duration_count[endpoint] += 1

    def record_prefill(self, status: int) -> None:
        self.prefill_total[str(status)] += 1

    def record_decode(self, status: int) -> None:
        self.decode_total[str(status)] += 1

    def generate_prometheus(self) -> str:
        lines: List[str] = [
            "# HELP vllm_proxy_requests_total Total HTTP requests processed by endpoint and status.",
            "# TYPE vllm_proxy_requests_total counter",
            ]
        for (endpoint, status), count in sorted(self.requests_total.items()):
            lines.append(f'vllm_proxy_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}')

        lines.extend(
                [
                    "# HELP vllm_proxy_prefill_requests_total Total prefill attempts by status.",
                    "# TYPE vllm_proxy_prefill_requests_total counter",
                    ], )
        for status, count in sorted(self.prefill_total.items()):
            lines.append(f'vllm_proxy_prefill_requests_total{{status="{status}"}} {count}')

        lines.extend(
                [
                    "# HELP vllm_proxy_decode_requests_total Total decode connection attempts by status.",
                    "# TYPE vllm_proxy_decode_requests_total counter",
                    ], )
        for status, count in sorted(self.decode_total.items()):
            lines.append(f'vllm_proxy_decode_requests_total{{status="{status}"}} {count}')

        lines.extend(
                [
                    "# HELP vllm_proxy_request_duration_seconds Summary of request processing duration in seconds.",
                    "# TYPE vllm_proxy_request_duration_seconds summary",
                    ], )
        for endpoint, duration_sum in sorted(self.request_duration_sum.items()):
            count = self.request_duration_count[endpoint]
            lines.append(f'vllm_proxy_request_duration_seconds_sum{{endpoint="{endpoint}"}} {duration_sum:.6f}')
            lines.append(f'vllm_proxy_request_duration_seconds_count{{endpoint="{endpoint}"}} {count}')

        return "\n".join(lines) + "\n"


metrics = ProxyMetrics()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM Disaggregated Router Proxy")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "4")))

    parser.add_argument(
            "--prefiller-hosts", type=str, nargs="+", default=os.getenv("PREFILLER_HOSTS", "localhost").split(","), )
    parser.add_argument(
            "--prefiller-ports", type=int, nargs="+",
            default=[int(p) for p in os.getenv("PREFILLER_PORTS", "8100").split(",")], )
    parser.add_argument(
            "--decoder-hosts", type=str, nargs="+", default=os.getenv("DECODER_HOSTS", "localhost").split(","), )
    parser.add_argument(
            "--decoder-ports", type=int, nargs="+",
            default=[int(p) for p in os.getenv("DECODER_PORTS", "8200").split(",")], )

    args, _ = parser.parse_known_args()

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


@functools.lru_cache(maxsize=1)
def get_args() -> argparse.Namespace:
    return parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown events."""
    args = get_args()
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}"
        app.state.prefill_clients.append(
                {
                    "client":
                        httpx.AsyncClient(
                            timeout=HTTP_TIMEOUT, limits=HTTP_LIMITS, base_url=prefiller_base_url, ), "host": host,
                    "port":                                                                                   port,
                    "id":
                        f"prefill-{i}-{host}:{port}",
                    }, )

    # Create decode clients
    for i, (host, port) in enumerate(args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}"
        app.state.decode_clients.append(
                {
                    "client":
                        httpx.AsyncClient(
                            timeout=HTTP_TIMEOUT, limits=HTTP_LIMITS, base_url=decoder_base_url, ), "host": host,
                    "port":                                                                                 port,
                    "id":
                        f"decode-{i}-{host}:{port}",
                    }, )

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    logger.info(
            f"Initialized {len(app.state.prefill_clients)} prefill clients and {len(app.state.decode_clients)} decode "
            f"clients.", )
    yield

    # Shutdown: Close all HTTP connections gracefully
    logger.info("Shutting down... Closing connection pools.")
    for client_info in app.state.prefill_clients + app.state.decode_clients:
        await client_info["client"].aclose()


app = FastAPI(lifespan=lifespan)


def get_next_client(app: FastAPI, service_type: str) -> Dict[str, Any]:
    """Get the next client in round-robin fashion."""
    if service_type == "prefill":
        return app.state.prefill_clients[next(app.state.prefill_iterator)]
    elif service_type == "decode":
        return app.state.decode_clients[next(app.state.decode_iterator)]
    raise ValueError(f"Unknown service type: {service_type}")


async def send_prefill_request_with_retries(
        app: FastAPI, endpoint: str, original_req_data: dict, request_id: str, headers: dict, ) -> Tuple[
    int, Any, Optional[int], Optional[int], Optional[Dict[str, Any]]]:
    # Deep copy to avoid mutating the original request dict across retries
    payload = copy.deepcopy(original_req_data)

    payload["kv_transfer_params"] = {
        "do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None, "remote_block_ids": None,
        "remote_host":      None, "remote_port": None,
        }
    payload["stream"] = False
    payload["max_tokens"] = 1

    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload.pop("stream_options", None)

    # These args are not supported for Prefill phase
    min_tokens = payload.pop("min_tokens", None)
    min_completion_tokens = payload.pop("min_completion_tokens", None)

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        client_info = get_next_client(app, "prefill")
        target_id = client_info["id"]

        try:
            logger.info(f"[{request_id}] Prefill attempt {attempt}/{MAX_RETRIES} -> {target_id}")
            response = await client_info["client"].post(endpoint, json=payload, headers=headers)

            # If client sent a bad request (4xx), return it immediately without retry.
            if 400 <= response.status_code < 500:
                metrics.record_prefill(response.status_code)
                logger.warning(f"[{request_id}] Prefill 4xx Error on {target_id}: {response.status_code}")
                return response.status_code, response.content, None, None, None

            response.raise_for_status()
            response_json = response.json()
            metrics.record_prefill(200)

            logger.info(f"[{request_id}] Prefill Success -> {target_id}")
            return 200, response_json, min_tokens, min_completion_tokens, client_info

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            last_error = e
            logger.warning(f"[{request_id}] Prefill failed on {target_id}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_SEC)

    logger.error(f"[{request_id}] All prefill retries exhausted.")
    if last_error:
        raise last_error
    raise RuntimeError("Prefill retries failed without specific exception")


async def stream_decode_with_retries(
        app: FastAPI, endpoint: str, req_data: dict, request_id: str, headers: dict, ) -> Tuple[
    int, Optional[bytes], Optional[httpx.Response]]:
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        client_info = get_next_client(app, "decode")
        target_id = client_info["id"]

        try:
            logger.info(f"[{request_id}] Decode connection attempt {attempt}/{MAX_RETRIES} -> {target_id}")
            # Build request manually so we can send with stream=True while catching network errors up-front
            req = client_info["client"].build_request("POST", endpoint, json=req_data, headers=headers)
            response = await client_info["client"].send(req, stream=True)

            if not response.is_success:
                error_content = await response.aread()
                await response.aclose()
                metrics.record_decode(response.status_code)
                logger.warning(f"[{request_id}] Decode error {response.status_code} on {target_id}")
                return response.status_code, error_content, None

            metrics.record_decode(200)
            logger.info(f"[{request_id}] Decode connection established -> {target_id}")
            return 200, None, response

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            last_error = e
            logger.warning(f"[{request_id}] Decode connection failed on {target_id}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_SEC)

    logger.error(f"[{request_id}] All decode retries exhausted.")
    if last_error:
        raise last_error
    raise RuntimeError("Decode retries failed without specific exception")


async def _handle_completions(api: str, request: Request):
    start_time = time.perf_counter()
    # Unique ID for tracing logs
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4())[:12])

    try:
        req_data = await request.json()
    except Exception:
        metrics.record_request(api, 400, time.perf_counter() - start_time)
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload"})

    model = req_data.get("model", "unknown")
    logger.info(f"[{request_id}] -> New request to {api} for model '{model}'")

    headers = {
        "X-Request-Id": request_id,
        }
    if request.client and request.client.host:
        headers["X-Forwarded-For"] = request.client.host

    try:
        # ==========================================
        # STEP 1: PREFILL PHASE
        # ==========================================
        p_status, p_data, min_t, min_comp_t, _ = await send_prefill_request_with_retries(
                request.app, api, req_data, request_id, headers, )

        # Fast exit if proxy received non-200 error
        if p_status != 200:
            metrics.record_request(api, p_status, time.perf_counter() - start_time)
            return Response(status_code=p_status, content=p_data, media_type="application/json")

        # ==========================================
        # STEP 2: PREPARE DECODE PAYLOAD
        # ==========================================
        decode_req_data = copy.deepcopy(req_data)

        kv_transfer_params = p_data.get("kv_transfer_params") if isinstance(p_data, dict) else None
        if not kv_transfer_params:
            logger.error(f"[{request_id}] Prefill response missing required 'kv_transfer_params': {p_data}")
            metrics.record_request(api, 502, time.perf_counter() - start_time)
            return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "Prefill worker did not return kv_transfer_params"}}, )

        decode_req_data["kv_transfer_params"] = kv_transfer_params

        if min_t is not None:
            decode_req_data["min_tokens"] = min_t
        if min_comp_t is not None:
            decode_req_data["min_completion_tokens"] = min_comp_t

        # ==========================================
        # STEP 3: DECODE PHASE
        # ==========================================
        d_status, d_error, d_response = await stream_decode_with_retries(
                request.app, api, decode_req_data, request_id, headers, )

        if d_status != 200 or d_response is None:
            metrics.record_request(api, d_status, time.perf_counter() - start_time)
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

        metrics.record_request(api, d_response.status_code, time.perf_counter() - start_time)
        return StreamingResponse(
                generate_stream(), media_type=d_response.headers.get("content-type", "application/json"),
                status_code=d_response.status_code, )

    except Exception as e:
        logger.error(f"[{request_id}] Unhandled internal proxy error:\n{traceback.format_exc()}")
        metrics.record_request(api, 500, time.perf_counter() - start_time)
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
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    headers = {"X-Request-Id": request_id}

    for attempt in range(1, MAX_RETRIES + 1):
        # We can just fetch this from the prefill pool, as it will mirror the model loaded
        client_info = get_next_client(request.app, "prefill")
        target_id = client_info["id"]

        try:
            logger.info(f"[{request_id}] Fetching models attempt {attempt}/{MAX_RETRIES} -> {target_id}")
            response = await client_info["client"].get("/v1/models", headers=headers, timeout=5.0)

            response.raise_for_status()
            return JSONResponse(status_code=response.status_code, content=response.json())

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
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
        "status":           "ok", "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
        }


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    return Response(content=metrics.generate_prometheus(), media_type="text/plain; version=0.0.4; charset=utf-8")


if __name__ == "__main__":
    args = get_args()

    # Pass args via environment variables so child processes inherit them cleanly
    os.environ["HOST"] = str(args.host)
    os.environ["PORT"] = str(args.port)
    os.environ["WORKERS"] = str(args.workers)
    os.environ["PREFILLER_HOSTS"] = ",".join(args.prefiller_hosts)
    os.environ["PREFILLER_PORTS"] = ",".join(map(str, args.prefiller_ports))
    os.environ["DECODER_HOSTS"] = ",".join(args.decoder_hosts)
    os.environ["DECODER_PORTS"] = ",".join(map(str, args.decoder_ports))

    import uvicorn

    # Pass import string "toy_proxy_server:app" for multi-worker support in uvicorn
    uvicorn.run(
            "toy_proxy_server:app", host=args.host, port=args.port, workers=args.workers, log_level="warning", )
