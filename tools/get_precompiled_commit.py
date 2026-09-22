#!/usr/bin/env python3
"""Finds the most recent commit in git history that has precompiled wheels available."""

import concurrent.futures
import os
import re
import subprocess
import urllib.request


def detect_variant() -> str:
    if os.getenv("VLLM_PRECOMPILED_WHEEL_VARIANT"):
        return os.getenv("VLLM_PRECOMPILED_WHEEL_VARIANT")
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        m = re.search(r"CUDA Version:\s*(\d+)", out.stdout)
        if m:
            major = int(m.group(1))
            return "cu130" if major >= 13 else "cu129"
    except Exception:
        pass
    return "cu129"


def check_commit(commit: str, variant: str) -> str | None:
    url = f"https://wheels.vllm.ai/{commit}/{variant}/vllm/metadata.json"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return commit
    except Exception:
        pass
    return None


def main() -> None:
    explicit = os.getenv("VLLM_PRECOMPILED_WHEEL_COMMIT")
    if explicit and len(explicit) == 40:
        print(explicit)
        return

    variant = detect_variant()
    try:
        commits = (
            subprocess.check_output(
                ["git", "rev-list", "HEAD", "-n", "50"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .split()
        )
    except Exception:
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(
            executor.map(lambda c: check_commit(c, variant), commits)
        )

    found = next((c for c in results if c), "")
    if found:
        print(found)


if __name__ == "__main__":
    main()
