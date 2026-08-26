import json
import time
import urllib.parse
import urllib.request
import gradio as gr


def send_http_request(
    url: str,
    endpoint: str,
    method: str = "POST",
    params: dict | None = None,
    json_body: dict | None = None,
    timeout: float = 10.0,
) -> dict:
    base = url.rstrip("/")
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    target_url = f"{base}{endpoint}{query}"
    start = time.perf_counter()

    headers = {}
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(target_url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            raw = resp.read().decode("utf-8")
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = raw
            return {
                "status": resp.status,
                "latency_ms": elapsed_ms,
                "data": parsed,
                "url": target_url,
            }
    except urllib.error.HTTPError as e:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        try:
            body = e.read().decode("utf-8")
            data = json.loads(body)
        except Exception:
            data = str(e)
        return {
            "error": f"HTTP {e.code}: {e.reason}",
            "status": e.code,
            "latency_ms": elapsed_ms,
            "data": data,
            "url": target_url,
        }
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "error": str(e),
            "latency_ms": elapsed_ms,
            "url": target_url,
        }


# ==============================================================================
# Actions
# ==============================================================================


def emergency_unfreeze(url: str) -> str:
    """1-Click Unfreeze Sequence: Pause(abort) -> Reset Cache(all) -> Resume."""
    if not url.strip():
        return json.dumps({"error": "Target URL is empty"}, indent=2)

    log = {}
    # 1. Pause with abort to drop stuck requests
    log["1_pause_abort"] = send_http_request(url, "/pause", method="POST", params={"mode": "abort"})
    # 2. Reset prefix cache (including connector external cache)
    log["2_reset_prefix_cache"] = send_http_request(
        url,
        "/reset_prefix_cache",
        method="POST",
        params={"reset_running_requests": "true", "reset_external": "true"},
    )
    # 3. Optional reset mm and encoder cache
    send_http_request(url, "/reset_mm_cache", method="POST")
    send_http_request(url, "/reset_encoder_cache", method="POST")
    # 4. Resume generation
    log["3_resume"] = send_http_request(url, "/resume", method="POST")

    return json.dumps({"action": "Emergency Full Unfreeze", "target": url, "sequence": log}, indent=2)


def reset_prefix_cache_action(url: str, reset_running: bool, reset_external: bool) -> str:
    res = send_http_request(
        url,
        "/reset_prefix_cache",
        method="POST",
        params={
            "reset_running_requests": str(reset_running).lower(),
            "reset_external": str(reset_external).lower(),
        },
    )
    return json.dumps(res, indent=2)


def abort_all_requests_action(url: str) -> str:
    res = send_http_request(url, "/abort_requests", method="POST", json_body={})
    return json.dumps(res, indent=2)


def pause_action(url: str, mode: str) -> str:
    res = send_http_request(url, "/pause", method="POST", params={"mode": mode})
    return json.dumps(res, indent=2)


def resume_action(url: str) -> str:
    res = send_http_request(url, "/resume", method="POST")
    return json.dumps(res, indent=2)


def check_pause_status(url: str) -> str:
    res = send_http_request(url, "/is_paused", method="GET")
    return json.dumps(res, indent=2)


def sleep_action(url: str, level: int, mode: str) -> str:
    res = send_http_request(url, "/sleep", method="POST", params={"level": str(level), "mode": mode})
    return json.dumps(res, indent=2)


def wake_up_action(url: str) -> str:
    res = send_http_request(url, "/wake_up", method="POST")
    return json.dumps(res, indent=2)


def check_sleep_status(url: str) -> str:
    res = send_http_request(url, "/is_sleeping", method="GET")
    return json.dumps(res, indent=2)


def get_diagnostics(url: str) -> str:
    if not url.strip():
        return json.dumps({"error": "Target URL is empty"}, indent=2)

    report = {
        "health": send_http_request(url, "/health", method="GET"),
        "is_paused": send_http_request(url, "/is_paused", method="GET"),
        "is_sleeping": send_http_request(url, "/is_sleeping", method="GET"),
        "load": send_http_request(url, "/load", method="GET"),
        "version": send_http_request(url, "/version", method="GET"),
        "server_info": send_http_request(url, "/server_info", method="GET", params={"config_format": "json"}),
    }
    return json.dumps(report, indent=2)


def profile_action(url: str, action: str) -> str:
    endpoint = "/start_profile" if action == "start" else "/stop_profile"
    res = send_http_request(url, endpoint, method="POST")
    return json.dumps(res, indent=2)


def collective_rpc_action(url: str, method_name: str, args_json: str, kwargs_json: str) -> str:
    try:
        args = json.loads(args_json) if args_json.strip() else []
        kwargs = json.loads(kwargs_json) if kwargs_json.strip() else {}
    except Exception as e:
        return json.dumps({"error": f"JSON parse error: {e}"}, indent=2)

    payload = {"method": method_name, "args": args, "kwargs": kwargs}
    res = send_http_request(url, "/collective_rpc", method="POST", json_body=payload)
    return json.dumps(res, indent=2)


# ==============================================================================
# Gradio UI
# ==============================================================================

with gr.Blocks(title="vLLM Dev & Maintenance Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 vLLM Dev & Maintenance Operations Center")
    gr.Markdown(
        "Manage disaggregated prefill/decode instances, unfreeze deadlocks, clear KV caches, "
        "and control engine lifecycles using vLLM dev/admin endpoints."
    )

    with gr.Row():
        selected_url = gr.Textbox(
            label="🎯 Target vLLM Instance URL",
            value="http://127.0.0.1:8000",
            placeholder="http://<host>:<port>",
            scale=3,
        )
        with gr.Column(scale=1):
            with gr.Row():
                btn_set_prefill = gr.Button("📍 Prefill (8000)", size="sm")
                btn_set_decode = gr.Button("📍 Decode (8001)", size="sm")

    btn_set_prefill.click(lambda: "http://127.0.0.1:8000", outputs=selected_url)
    btn_set_decode.click(lambda: "http://127.0.0.1:8001", outputs=selected_url)

    with gr.Tabs():
        # --- TAB 1: EMERGENCY RECOVERY ---
        with gr.Tab("🚨 Emergency Recovery"):
            gr.Markdown(
                "### 🔄 Recommended: Sequence Unfreeze (`/pause?mode=abort` ➔ `/reset_prefix_cache` ➔ `/resume`)\n"
                "Instantly clears blocked queues, resets stuck RDMA/NIXL transfers, and restores engine throughput in < 1 second."
            )
            with gr.Row():
                btn_unfreeze_single = gr.Button("⚡ Emergency Unfreeze (Target Instance)", variant="primary", scale=2)
                btn_unfreeze_both = gr.Button("🔥 Full Reset Both (Prefill + Decode)", variant="stop", scale=2)

            gr.Markdown("### 🧹 Granular Cache & Request Reset")
            with gr.Row():
                reset_running_chk = gr.Checkbox(label="Preempt Running Requests", value=True)
                reset_external_chk = gr.Checkbox(label="Reset External / Connector Cache", value=True)
            with gr.Row():
                btn_flush_cache = gr.Button("🧹 Reset Prefix Cache (/reset_prefix_cache)")
                btn_abort_all = gr.Button("🛑 Abort All Requests (/abort_requests)")

        # --- TAB 2: FLOW CONTROL ---
        with gr.Tab("⏸️ Flow Control & Pausing"):
            gr.Markdown("Control request ingestion and token generation without restarting.")
            with gr.Row():
                pause_mode = gr.Radio(
                    label="Pause Mode",
                    choices=["abort", "wait", "keep"],
                    value="abort",
                    info="'abort': drop in-flight, 'wait': drain cleanly, 'keep': freeze in queue",
                )
            with gr.Row():
                btn_pause = gr.Button("⏸️ Pause Generation (/pause)", variant="secondary")
                btn_resume = gr.Button("▶️ Resume Generation (/resume)", variant="primary")
                btn_check_paused = gr.Button("🔍 Check Is Paused (/is_paused)")

        # --- TAB 3: POWER / SLEEP ---
        with gr.Tab("💤 Sleep & Memory Offload"):
            gr.Markdown("Offload GPU allocations to CPU/host memory when idle.")
            with gr.Row():
                sleep_level = gr.Slider(label="Sleep Level", minimum=1, maximum=2, step=1, value=1)
                sleep_mode = gr.Dropdown(label="Sleep Mode", choices=["abort", "wait"], value="abort")
            with gr.Row():
                btn_sleep = gr.Button("🌙 Sleep Engine (/sleep)", variant="secondary")
                btn_wake = gr.Button("☀️ Wake Up Engine (/wake_up)", variant="primary")
                btn_check_sleep = gr.Button("🔍 Check Is Sleeping (/is_sleeping)")

        # --- TAB 4: DIAGNOSTICS & SYSTEM INFO ---
        with gr.Tab("📊 Diagnostics & Server Info"):
            gr.Markdown("Inspect engine health, load, active configs, and environment variables.")
            with gr.Row():
                btn_run_diag = gr.Button("🔍 Fetch Full Diagnostics (/server_info, /load, /health)", variant="primary")

        # --- TAB 5: ADVANCED / PROFILER & RPC ---
        with gr.Tab("🔬 Profiling & Collective RPC"):
            gr.Markdown("Run PyTorch CUDA profiler or send RPC calls across distributed TP/PP ranks.")
            with gr.Row():
                btn_start_prof = gr.Button("🔴 Start Profiler (/start_profile)")
                btn_stop_prof = gr.Button("⏹️ Stop Profiler (/stop_profile)")

            gr.Markdown("#### Collective RPC (/collective_rpc)")
            rpc_method = gr.Textbox(label="Method Name", placeholder="e.g. reset_prefix_cache", value="")
            with gr.Row():
                rpc_args = gr.Textbox(label="Args (JSON Array)", placeholder="[]", value="[]")
                rpc_kwargs = gr.Textbox(label="Kwargs (JSON Object)", placeholder="{}", value="{}")
            btn_run_rpc = gr.Button("⚡ Execute Collective RPC")

    # Output Console
    gr.Markdown("### 📜 Operation Response")
    output_box = gr.Code(label="JSON Result", language="json", lines=18)

    # --- Bindings ---
    btn_unfreeze_single.click(emergency_unfreeze, inputs=[selected_url], outputs=[output_box])

    def unfreeze_both_fn(cur_url: str):
        p_res = emergency_unfreeze("http://127.0.0.1:8000")
        d_res = emergency_unfreeze("http://127.0.0.1:8001")
        return json.dumps({"Prefill": json.loads(p_res), "Decode": json.loads(d_res)}, indent=2)

    btn_unfreeze_both.click(unfreeze_both_fn, inputs=[selected_url], outputs=[output_box])
    btn_flush_cache.click(
        reset_prefix_cache_action,
        inputs=[selected_url, reset_running_chk, reset_external_chk],
        outputs=[output_box],
    )
    btn_abort_all.click(abort_all_requests_action, inputs=[selected_url], outputs=[output_box])

    btn_pause.click(pause_action, inputs=[selected_url, pause_mode], outputs=[output_box])
    btn_resume.click(resume_action, inputs=[selected_url], outputs=[output_box])
    btn_check_paused.click(check_pause_status, inputs=[selected_url], outputs=[output_box])

    btn_sleep.click(sleep_action, inputs=[selected_url, sleep_level, sleep_mode], outputs=[output_box])
    btn_wake.click(wake_up_action, inputs=[selected_url], outputs=[output_box])
    btn_check_sleep.click(check_sleep_status, inputs=[selected_url], outputs=[output_box])

    btn_run_diag.click(get_diagnostics, inputs=[selected_url], outputs=[output_box])

    btn_start_prof.click(lambda u: profile_action(u, "start"), inputs=[selected_url], outputs=[output_box])
    btn_stop_prof.click(lambda u: profile_action(u, "stop"), inputs=[selected_url], outputs=[output_box])

    btn_run_rpc.click(
        collective_rpc_action,
        inputs=[selected_url, rpc_method, rpc_args, rpc_kwargs],
        outputs=[output_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
