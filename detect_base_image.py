#!/usr/bin/env python3
"""Detect the right vllm/vllm-openai base image tag for this fork.

Algorithm:
  1. Find the merge-base of HEAD with <remote>/<branch> (default upstream/main).
  2. Fetch the list of published nightly tags for the chosen variant from
     Docker Hub.
  3. Walk backwards from the merge-base through upstream history and return
     the first commit SHA that has a matching tag.

The resulting tag is printed to stdout (so it is usable in shell
substitution: `BASE_TAG=$(python detect_base_image.py)`). Progress goes to
stderr.

Examples:
    python detect_base_image.py                       # print best cu130 tag
    python detect_base_image.py --variant ''          # default (non-cu) variant
    python detect_base_image.py --update              # rewrite the dockerfile
    python detect_base_image.py --build               # update + docker build
    python detect_base_image.py --push                # update + build + push
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "vllm/vllm-openai"
TAGS_URL = f"https://hub.docker.com/v2/repositories/{REPO}/tags/"
DOCKERFILE = Path(__file__).resolve().parent / "vllm-infomaniak.dockerfile"
DEFAULT_REGISTRY = "registry.infomaniak.com:443/r-and-d/ai/k8s-llm/vllm-openai"


def run(cmd: list[str]) -> str:
    return subprocess.run(
        cmd, capture_output=True, text=True, check=True
    ).stdout.strip()


def fetch_tag_shas(variant: str, max_pages: int = 5) -> dict[str, str]:
    """Return {commit_sha: full_tag_name} for nightly tags of the variant."""
    prefix = f"{variant}-nightly-" if variant else "nightly-"
    sha_re = re.compile(rf"^{re.escape(prefix)}([0-9a-f]{{40}})$")
    shas: dict[str, str] = {}
    url = f"{TAGS_URL}?name={prefix}&page_size=100"
    pages = 0
    while url and pages < max_pages:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
        for tag in data.get("results", []):
            m = sha_re.match(tag["name"])
            if m:
                shas[m.group(1)] = tag["name"]
        url = data.get("next")
        pages += 1
    return shas


def detect(
    variant: str,
    remote: str,
    branch: str,
    max_walk: int,
    fetch: bool,
) -> tuple[str, str]:
    """Return (sha, tag_name) for the best base image."""
    if fetch:
        print(f"Fetching {remote}...", file=sys.stderr)
        run(["git", "fetch", remote, "--quiet"])

    ref = f"{remote}/{branch}"
    merge_base = run(["git", "merge-base", "HEAD", ref])
    print(f"Merge-base with {ref}: {merge_base}", file=sys.stderr)

    print(f"Fetching Docker Hub tags for {REPO}...", file=sys.stderr)
    shas = fetch_tag_shas(variant)
    label = variant or "(no-variant)"
    print(f"  found {len(shas)} {label}-nightly tags", file=sys.stderr)
    if not shas:
        raise SystemExit(
            f"No tags found on Docker Hub for variant {label!r}. "
            f"Check that the variant prefix is correct."
        )

    revs = run(
        ["git", "rev-list", f"--max-count={max_walk}", merge_base]
    ).splitlines()

    for i, sha in enumerate(revs):
        if sha in shas:
            print(
                f"  matched {sha[:12]} after walking back {i} commit(s)",
                file=sys.stderr,
            )
            return sha, shas[sha]

    raise SystemExit(
        f"No matching tag found within {max_walk} commits of {merge_base}. "
        f"Try increasing --max-walk."
    )


def docker_build(image: str) -> None:
    cmd = ["docker", "build", "-f", DOCKERFILE.name, "-t", image, "."]
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True, cwd=DOCKERFILE.parent)


def docker_push(image: str) -> None:
    cmd = ["docker", "push", image]
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def update_dockerfile(variant: str, sha: str) -> None:
    if not DOCKERFILE.exists():
        raise SystemExit(f"Dockerfile not found: {DOCKERFILE}")
    text = DOCKERFILE.read_text()
    prefix = f"{variant}-nightly-" if variant else "nightly-"
    pattern = re.compile(rf"({re.escape(prefix)})[0-9a-f]{{40}}")
    new_text, n = pattern.subn(rf"\g<1>{sha}", text)
    if n == 0:
        raise SystemExit(
            f"Could not find any {prefix}<sha> reference in {DOCKERFILE.name}"
        )
    if new_text != text:
        DOCKERFILE.write_text(new_text)
        print(
            f"Updated {DOCKERFILE.name}: {n} occurrence(s) -> {prefix}{sha}",
            file=sys.stderr,
        )
    else:
        print(f"{DOCKERFILE.name} already on {prefix}{sha}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--variant",
        default="cu130",
        help="Image variant prefix (default: cu130). Use '' for the no-variant nightly.",
    )
    p.add_argument("--remote", default="upstream", help="Git remote name (default: upstream).")
    p.add_argument("--branch", default="main", help="Upstream branch name (default: main).")
    p.add_argument(
        "--max-walk",
        type=int,
        default=200,
        help="Max commits to walk back from the merge-base (default: 200).",
    )
    p.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip 'git fetch <remote>' before computing the merge-base.",
    )
    p.add_argument(
        "--update",
        action="store_true",
        help="Rewrite tag references in vllm-infomaniak.dockerfile in place.",
    )
    p.add_argument(
        "--build",
        action="store_true",
        help="Run 'docker build' with the detected tag (implies --update).",
    )
    p.add_argument(
        "--push",
        action="store_true",
        help="Run 'docker push' after building (implies --build).",
    )
    p.add_argument(
        "--registry",
        default=DEFAULT_REGISTRY,
        help=f"Image name (without tag) for build/push. Default: {DEFAULT_REGISTRY}",
    )
    args = p.parse_args()

    sha, tag = detect(
        variant=args.variant,
        remote=args.remote,
        branch=args.branch,
        max_walk=args.max_walk,
        fetch=not args.no_fetch,
    )

    do_build = args.build or args.push
    do_update = args.update or do_build
    if do_update:
        update_dockerfile(args.variant, sha)

    if do_build:
        image = f"{args.registry}:{tag}"
        docker_build(image)
        if args.push:
            docker_push(image)

    print(tag)


if __name__ == "__main__":
    main()