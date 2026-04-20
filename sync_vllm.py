#!/usr/bin/env python3
"""
Synchronize vLLM fork with upstream and update custom Dockerfiles with the latest
compatible nightly base image.

This script:
1. Syncs the local branch (default: main) with upstream.
2. Detects the best vllm/vllm-openai base image tag matching the merge-base.
3. Updates vllm-infomaniak.dockerfile and vllm-dev-infomaniak.dockerfile.
4. (Optional) Builds and pushes the updated images.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

# --- Configuration ---
UPSTREAM_URL = "https://github.com/vllm-project/vllm.git"
DOCKER_REPO = "vllm/vllm-openai"
TAGS_URL = f"https://hub.docker.com/v2/repositories/{DOCKER_REPO}/tags/"
DEFAULT_REGISTRY = "registry.infomaniak.com:443/r-and-d/ai/k8s-llm/vllm-openai"
DOCKERFILES = [
    "vllm-infomaniak.dockerfile", "vllm-dev-infomaniak.dockerfile",
    ]


# --- Utility Functions ---
def print_color(text: str, color: str) -> None:
    """Print colored text using ANSI codes."""
    colors = {
        "cyan": "\033[36m", "yellow": "\033[33m", "green": "\033[32m", "red": "\033[31m",
        }
    reset = "\033[0m"
    print(f"{colors.get(color, '')}{text}{reset}", file=sys.stderr)


def run_git(args: list[str], check: bool = True) -> str:
    """Run a git command and return its output."""
    result = subprocess.run(
            ["git"] + args, capture_output=True, text=True, check=False
            )
    if check and result.returncode != 0:
        print_color(f"Git command failed: git {' '.join(args)}", "red")
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip()


# --- Sync Logic ---
def check_upstream(remote_name: str) -> None:
    """Ensure the upstream remote is configured."""
    remotes = run_git(["remote", "-v"])
    if remote_name not in remotes:
        print_color(f"'{remote_name}' remote is not configured.", "yellow")
        print_color(f"Do you want to add it? (URL: {UPSTREAM_URL})", "yellow")
        response = input("Enter 'y' to proceed, or any other key to cancel: ")
        if response.lower() == "y":
            run_git(["remote", "add", remote_name, UPSTREAM_URL])
            print_color(f"✓ Remote '{remote_name}' added.", "green")
        else:
            print_color("Operation cancelled.", "yellow")
            sys.exit(0)


def get_branch_info(remote: str, branch: str) -> tuple[int, int]:
    """Get commit count ahead/behind relative to upstream."""
    try:
        ahead_behind = run_git(
                ["rev-list", "--left-right", f"{remote}/{branch}...HEAD", "--count"], check=False
                )
        if ahead_behind:
            parts = ahead_behind.strip().split()
            if len(parts) == 2:
                return int(parts[1]), int(parts[0])
    except Exception:
        pass
    return 0, 0


def sync_fork(remote: str, branch: str) -> None:
    """Sync the local branch with upstream."""
    print_color(f"=== Syncing fork with {remote}/{branch} ===", "cyan")

    check_upstream(remote)

    print_color(f"Step 1: Fetching {remote} changes...", "yellow")
    run_git(["fetch", remote])

    ahead, behind = get_branch_info(remote, branch)
    print_color(f"Status: {ahead} ahead, {behind} behind {remote}/{branch}", "yellow")

    if behind == 0:
        print_color("✓ Branch is already up to date with upstream.", "green")
        return

    print_color(f"Step 2: Checking out {branch}...", "yellow")
    run_git(["checkout", branch])

    print_color(f"Step 3: Merging {remote}/{branch}...", "yellow")
    run_git(["merge", f"{remote}/{branch}", "--no-edit"])

    print_color(f"Step 4: Pushing to origin/{branch}...", "yellow")
    run_git(["push", "origin", branch])

    print_color("✓ Sync completed successfully.", "green")


# --- Image Detection Logic ---
def fetch_tag_shas(variant: str, max_pages: int = 5) -> dict[str, str]:
    """Return {commit_sha: full_tag_name} for nightly tags from Docker Hub."""
    prefix = f"{variant}-nightly-" if variant else "nightly-"
    sha_re = re.compile(rf"^{re.escape(prefix)}([0-9a-f]{{40}})$")
    shas: dict[str, str] = {}
    url = f"{TAGS_URL}?name={prefix}&page_size=100"
    pages = 0
    while url and pages < max_pages:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.load(resp)
            for tag in data.get("results", []):
                m = sha_re.match(tag["name"])
                if m:
                    shas[m.group(1)] = tag["name"]
            url = data.get("next")
            pages += 1
        except Exception as e:
            print_color(f"Warning: Failed to fetch tags from Docker Hub: {e}", "yellow")
            break
    return shas


def detect_base_image(variant: str, remote: str, branch: str, max_walk: int) -> tuple[str, str]:
    """Find the best base image tag matching the merge-base."""
    print_color(f"=== Detecting best base image for {variant} ===", "cyan")

    merge_base = run_git(["merge-base", "HEAD", f"{remote}/{branch}"])
    print(f"Merge-base with {remote}/{branch}: {merge_base}", file=sys.stderr)

    print(f"Fetching Docker Hub tags for {DOCKER_REPO}...", file=sys.stderr)
    shas = fetch_tag_shas(variant)
    if not shas:
        print_color(f"Error: No tags found for variant '{variant}'.", "red")
        sys.exit(1)

    revs = run_git(["rev-list", f"--max-count={max_walk}", merge_base]).splitlines()
    for i, sha in enumerate(revs):
        if sha in shas:
            print_color(f"✓ Matched {sha[:12]} after walking back {i} commits.", "green")
            return sha, shas[sha]

    print_color(f"Error: No matching tag found within {max_walk} commits.", "red")
    sys.exit(1)


def update_dockerfiles(variant: str, sha: str) -> None:
    """Update tag references in all configured Dockerfiles."""
    prefix = f"{variant}-nightly-" if variant else "nightly-"
    pattern = re.compile(rf"({re.escape(prefix)})[0-9a-f]{{40}}")

    for df_name in DOCKERFILES:
        df_path = Path(df_name)
        if not df_path.exists():
            print_color(f"Warning: {df_name} not found, skipping.", "yellow")
            continue

        text = df_path.read_text()
        new_text, n = pattern.subn(rf"\g<1>{sha}", text)

        if n == 0:
            print_color(f"Warning: No {prefix}<sha> found in {df_name}.", "yellow")
            continue

        if new_text != text:
            df_path.write_text(new_text)
            print_color(f"✓ Updated {df_name}: {n} occurrence(s) -> {prefix}{sha}", "green")
        else:
            print(f"{df_name} is already up to date.", file=sys.stderr)


def docker_build_and_push(image: str, dockerfile: str, push: bool) -> None:
    """Build and optionally push a docker image."""
    print_color(f"=== Building {image} ===", "cyan")
    cmd = ["docker", "build", "-f", dockerfile, "-t", image, "."]
    subprocess.run(cmd, check=True)
    if push:
        print_color(f"=== Pushing {image} ===", "cyan")
        subprocess.run(["docker", "push", image], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Sync args
    p.add_argument("--remote", default="upstream", help="Upstream remote name (default: upstream)")
    p.add_argument("--branch", default="main", help="Target branch (default: main)")
    p.add_argument("--skip-sync", action="store_true", help="Skip syncing with upstream")

    # Detection args
    p.add_argument("--variant", default="cu130", help="Image variant prefix (default: cu130)")
    p.add_argument("--max-walk", type=int, default=200, help="Max commits to walk back (default: 200)")
    p.add_argument("--no-update", action="store_true", help="Don't update Dockerfiles")

    # Build args
    p.add_argument("--build", action="store_true", help="Build docker images after update")
    p.add_argument("--push", action="store_true", help="Push docker images after build")
    p.add_argument("--registry", default=DEFAULT_REGISTRY, help=f"Image registry (default: {DEFAULT_REGISTRY})")

    args = p.parse_args()

    # 1. Sync
    if not args.skip_sync:
        sync_fork(args.remote, args.branch)
        print()

    # 2. Detect
    sha, tag = detect_base_image(args.variant, args.remote, args.branch, args.max_walk)
    print(f"Target Tag: {tag}")
    print()

    # 3. Update
    if not args.no_update:
        update_dockerfiles(args.variant, sha)
        print()

    # 4. Build / Push
    if args.build or args.push:
        for df in DOCKERFILES:
            # For simplicity, we use the same tag for our images as the base image
            image_name = f"{args.registry}:{tag}"
            if "dev" in df:
                image_name += "-dev"
            docker_build_and_push(image_name, df, args.push)


if __name__ == "__main__":
    main()
