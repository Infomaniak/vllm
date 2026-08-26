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
# Collective RPC Preset Definitions
# ==============================================================================

RPC_PRESETS = {
    "-- Select a Preset --": {
        "method": "",
        "args": "[]",
        "kwargs": "{}",
        "desc": "Select a preset above to populate method, args, and kwargs automatically, or type custom values below.",
    },
    "1. 🩺 Worker & NCCL Health Check": {
        "method": "check_health",
        "args": "[]",
        "kwargs": "{}",
        "desc": "Runs an internal health check / barrier across all TP/PP worker ranks to verify no worker has crashed or hung on NCCL.",
    },
    "2. ⚡ Execute Dummy Forward Pass": {
        "method": "execute_dummy_batch",
        "args": "[]",
        "kwargs": "{}",
        "desc": "Forces all GPU workers to execute a dummy forward pass through CUDA kernels to verify GPU execution without a client request.",
    },
    "3. 📋 List Loaded LoRA Adapters": {
        "method": "list_loras",
        "args": "[]",
        "kwargs": "{}",
        "desc": "Queries all active LoRA IDs currently loaded in GPU memory across all TP ranks.",
    },
    "4. 🗑️ Remove / Unload LoRA Adapter": {
        "method": "remove_lora",
        "args": "[1]",
        "kwargs": "{}",
        "desc": "Unloads the specified LoRA adapter ID (e.g. ID 1) from all worker ranks to reclaim GPU memory.",
    },
    "5. 📌 Pin LoRA Adapter in Memory": {
        "method": "pin_lora",
        "args": "[1]",
        "kwargs": "{}",
        "desc": "Pins the specified LoRA adapter ID (e.g. ID 1) in GPU memory across workers to prevent eviction.",
    },
    "6. 📏 Update Max Model Context Length": {
        "method": "update_max_model_len",
        "args": "[8192]",
        "kwargs": "{}",
        "desc": "Dynamically updates the maximum context length across all workers without restarting the server.",
    },
    "7. ⏱️ Multi-Modal Encoder Timing Stats": {
        "method": "get_encoder_timing_stats",
        "args": "[]",
        "kwargs": "{}",
        "desc": "Retrieves latency and timing statistics for vision/audio multimodal encoders across workers.",
    },
    "8. 🔄 Hot-Reload Model Weights": {
        "method": "reload_weights",
        "args": "[]",
        "kwargs": "{}",
        "desc": "Hot-reloads model weights from disk or memory buffers across all worker processes without restarting.",
    },
    "9. 💾 Save Sharded State to Disk": {
        "method": "save_sharded_state",
        "args": '["/tmp/sharded_checkpoint"]',
        "kwargs": "{}",
        "desc": "Dumps the current in-memory sharded model weights across all TP ranks to the specified directory.",
    },
    "10. 🔴 Named Profiler Trace (Start)": {
        "method": "profile",
        "args": "[]",
        "kwargs": '{"is_start": true, "profile_prefix": "debug_prefill_trace"}',
        "desc": "Starts a named PyTorch/CUDA profiler trace across all distributed workers.",
    },
    "11. ⏹️ Named Profiler Trace (Stop)": {
        "method": "profile",
        "args": "[]",
        "kwargs": '{"is_start": false}',
        "desc": "Flushes and stops the active named profiler trace across all distributed workers.",
    },
    "12. 🧹 Reset Multi-Modal Cache": {
        "method": "reset_mm_cache",
        "args": "[]",
        "kwargs": "{}",
        "desc": "Clears the vision/audio multimodal embedding cache across all workers.",
    },
    "13. 🧹 Reset Encoder Cache": {
        "method": "reset_encoder_cache",
        "args": "[]",
        "kwargs": "{}",
        "desc": "Clears the encoder cache across all workers for cross-attention models.",
    },
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
    if not method_name.strip():
        return json.dumps({"error": "Method Name cannot be empty"}, indent=2)
    try:
        args = json.loads(args_json) if args_json.strip() else []
        kwargs = json.loads(kwargs_json) if kwargs_json.strip() else {}
    except Exception as e:
        return json.dumps({"error": f"JSON parse error: {e}"}, indent=2)

    payload = {"method": method_name.strip(), "args": args, "kwargs": kwargs}
    res = send_http_request(url, "/collective_rpc", method="POST", json_body=payload)
    return json.dumps(res, indent=2)


def unfreeze_both_fn(p_url: str, d_url: str) -> str:
    p_res = emergency_unfreeze(p_url) if p_url.strip() else json.dumps({"status": "skipped (no Prefill URL)"})
    d_res = emergency_unfreeze(d_url) if d_url.strip() else json.dumps({"status": "skipped (no Decode URL)"})
    try:
        p_obj = json.loads(p_res)
    except Exception:
        p_obj = p_res
    try:
        d_obj = json.loads(d_res)
    except Exception:
        d_obj = d_res
    return json.dumps({"Prefill": p_obj, "Decode": d_obj}, indent=2)


# ==============================================================================
# Gradio UI
# ==============================================================================

with gr.Blocks(title="vLLM Dev & Maintenance Dashboard") as demo:
    gr.Markdown("# 🚀 vLLM Dev & Maintenance Operations Center")
    gr.Markdown(
        "Manage disaggregated prefill/decode instances, unfreeze deadlocks, clear KV caches, "
        "and control engine lifecycles using vLLM dev/admin endpoints."
    )

    with gr.Row():
        prefill_url = gr.Textbox(
            label="📍 Prefill Instance URL",
            value="http://127.0.0.1:8010",
            placeholder="http://<prefill-host>:<port>",
            scale=1,
        )
        decode_url = gr.Textbox(
            label="📍 Decode Instance URL",
            value="http://127.0.0.1:8020",
            placeholder="http://<decode-host>:<port>",
            scale=1,
        )

    with gr.Row():
        target_choice = gr.Radio(
            label="🎯 Target Instance for Operations",
            choices=["Prefill", "Decode", "Custom"],
            value="Prefill",
            scale=1,
        )
        selected_url = gr.Textbox(
            label="Active Operation URL",
            value="http://127.0.0.1:8000",
            placeholder="http://<host>:<port>",
            scale=2,
        )

    def on_target_choice_changed(choice: str, p_url: str, d_url: str, current_sel: str) -> str:
        if choice == "Prefill":
            return p_url
        elif choice == "Decode":
            return d_url
        else:
            return current_sel

    def on_prefill_url_changed(p_url: str, choice: str) -> str:
        if choice == "Prefill":
            return p_url
        return gr.skip()

    def on_decode_url_changed(d_url: str, choice: str) -> str:
        if choice == "Decode":
            return d_url
        return gr.skip()

    target_choice.change(
        on_target_choice_changed,
        inputs=[target_choice, prefill_url, decode_url, selected_url],
        outputs=[selected_url],
    )
    prefill_url.change(
        on_prefill_url_changed,
        inputs=[prefill_url, target_choice],
        outputs=[selected_url],
    )
    decode_url.change(
        on_decode_url_changed,
        inputs=[decode_url, target_choice],
        outputs=[selected_url],
    )

    with gr.Tabs():
        # --- TAB 1: EMERGENCY RECOVERY ---
        with gr.Tab("🚨 Emergency Recovery"):
            gr.Markdown(
                "### 🔄 Sequence Unfreeze (`/pause?mode=abort` ➔ `/reset_prefix_cache` ➔ `/resume`)\n"
                "Instantly clears blocked queues, resets stuck RDMA/NIXL transfers, and restores engine throughput in < 1 second."
            )
            with gr.Row():
                btn_unfreeze_single = gr.Button("⚡ Emergency Unfreeze (Active Target)", variant="primary", scale=2)
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

            gr.Markdown("---")
            gr.Markdown("### ⚡ Collective RPC Runner (`/collective_rpc`)")
            gr.Markdown("Execute arbitrary worker methods simultaneously across all distributed TP/PP worker processes.")

            preset_dropdown = gr.Dropdown(
                label="📦 Quick Presets (Select a pre-configured RPC command)",
                choices=list(RPC_PRESETS.keys()),
                value="-- Select a Preset --",
            )
            preset_desc = gr.Markdown(RPC_PRESETS["-- Select a Preset --"]["desc"])

            rpc_method = gr.Textbox(label="Method Name", placeholder="e.g. check_health", value="")
            with gr.Row():
                rpc_args = gr.Textbox(label="Args (JSON Array)", placeholder="[]", value="[]")
                rpc_kwargs = gr.Textbox(label="Kwargs (JSON Object)", placeholder="{}", value="{}")
            btn_run_rpc = gr.Button("⚡ Execute Collective RPC", variant="primary")

            def on_preset_selected(preset_key: str):
                item = RPC_PRESETS.get(preset_key, RPC_PRESETS["-- Select a Preset --"])
                return item["method"], item["args"], item["kwargs"], f"💡 **Details:** {item['desc']}"

            preset_dropdown.change(
                on_preset_selected,
                inputs=[preset_dropdown],
                outputs=[rpc_method, rpc_args, rpc_kwargs, preset_desc],
            )

    # Output Console
    gr.Markdown("### 📜 Operation Response")
    output_box = gr.Code(label="JSON Result", language="json", lines=18)

    # --- Bindings ---
    btn_unfreeze_single.click(emergency_unfreeze, inputs=[selected_url], outputs=[output_box])
    btn_unfreeze_both.click(unfreeze_both_fn, inputs=[prefill_url, decode_url], outputs=[output_box])

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
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7860)
