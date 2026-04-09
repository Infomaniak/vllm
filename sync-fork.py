#!/usr/bin/env python3
"""
Python script to sync fork main branch with upstream.
This script fetches upstream changes and merges them into the local main branch.
"""

import subprocess
import sys
import argparse

UPSTREAM_URL = "https://github.com/vllm-project/vllm.git"


def print_color(text, color):
    """Print colored text using ANSI codes."""
    colors = {
        "cyan": "\033[36m",
        "yellow": "\033[33m",
        "green": "\033[32m",
        "red": "\033[31m",
    }
    reset = "\033[0m"
    print(f"{colors.get(color, '')}{text}{reset}")


def run_git_command(args, check=True):
    """Run a git command and return success status."""
    try:
        result = subprocess.run(
            ["git"] + args, capture_output=True, text=True, check=False
        )
        if check and result.returncode != 0:
            return False
        return True
    except FileNotFoundError:
        return False


def get_git_output(args):
    """Run a git command and return the output."""
    try:
        result = subprocess.run(
            ["git"] + args, capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            return result.stdout
        return ""
    except FileNotFoundError:
        return ""


def check_git_repo():
    """Check if we're in a git repository."""
    if not run_git_command(["rev-parse", "--git-dir"]):
        print_color(
            "Error: Not a git repository. Please run this script from within your git repository.",
            "red",
        )
        sys.exit(1)


def check_upstream():
    """Check if 'upstream' remote exists, add it if not."""
    remotes = get_git_output(["remote", "-v"])

    if "upstream" not in remotes:
        print_color("'upstream' remote is not configured.", "yellow")
        print_color(f"Do you want to add it? (URL: {UPSTREAM_URL})", "yellow")
        response = input(
            "Enter 'y' to proceed with adding upstream, or any other key to cancel: "
        )

        if response.lower() == "y":
            print_color("Adding upstream remote...", "yellow")
            if not run_git_command(["remote", "add", "upstream", UPSTREAM_URL]):
                print_color("Error: Failed to add upstream remote.", "red")
                sys.exit(1)
            print_color("✓ Upstream remote added successfully", "green")
        else:
            print_color("Operation cancelled. Upstream remote not added.", "yellow")
            sys.exit(0)

    print()


def get_git_status():
    """Get git status output."""
    output = get_git_output(["status", "--short"])
    return output.strip() if output else ""


def get_branch_info(branch):
    """Get information about local vs upstream branch."""
    try:
        # Get commit count ahead/behind
        ahead_behind = get_git_output(
            ["rev-list", "--left-right", f"upstream/{branch}...HEAD", "--count"]
        )
        if ahead_behind:
            parts = ahead_behind.strip().split()
            if len(parts) == 2:
                behind, ahead = int(parts[0]), int(parts[1])
                return ahead, behind
        return 0, 0
    except:
        return 0, 0


def print_status_summary(branch):
    """Print a summary of the current git status."""
    status = get_git_status()
    ahead, behind = get_branch_info(branch)

    if status:
        print_color("Current changes:", "yellow")
        print(status)
        print()

    if ahead > 0 or behind > 0:
        print_color("Branch status:", "yellow")
        if behind > 0:
            print(f"  {behind} commits behind upstream/{branch}")
        if ahead > 0:
            print(f"  {ahead} commits ahead of upstream/{branch}")
        print()


def sync_fork(branch):
    """Main sync logic."""
    print_color("=== Syncing fork with upstream ===", "cyan")
    print()

    # Check git repo and upstream
    check_git_repo()
    check_upstream()

    # Show initial status
    print_color(f"On branch {branch}", "yellow")
    print_status_summary(branch)

    # Step 1: Fetch upstream changes
    print_color("Step 1: Fetching upstream changes...", "yellow")
    if not run_git_command(["fetch", "upstream"]):
        print_color("Error: Failed to fetch from upstream.", "red")
        sys.exit(1)
    print_color("✓ Upstream changes fetched successfully", "green")
    print()

    # Show what will be merged
    ahead, behind = get_branch_info(branch)
    if behind > 0:
        print_color(f"Found {behind} commit(s) to sync from upstream/{branch}", "green")
        print()

    # Step 2: Checkout the target branch
    print_color(f"Step 2: Checking out {branch} branch...", "yellow")
    if not run_git_command(["checkout", branch]):
        print_color(f"Error: Failed to checkout {branch} branch.", "red")
        sys.exit(1)
    print_color(f"✓ Checked out {branch} branch", "green")
    print()

    # Step 3: Merge upstream changes
    print_color(f"Step 3: Merging upstream/{branch} into {branch}...", "yellow")
    if not run_git_command(["merge", f"upstream/{branch}"]):
        print_color("Error: Merge failed. Please resolve conflicts manually.", "red")
        sys.exit(1)
    print_color(f"✓ Merged upstream/{branch} successfully", "green")
    print()

    # Step 4: Push to origin
    print_color(f"Step 4: Pushing to origin/{branch}...", "yellow")
    if not run_git_command(["push", "origin", branch]):
        print_color("Error: Failed to push to origin.", "red")
        sys.exit(1)
    print_color(f"✓ Pushed to origin/{branch} successfully", "green")
    print()

    # Show final status
    print_color("Current status:", "yellow")
    print_status_summary(branch)

    print_color("=== Sync completed successfully! ===", "cyan")
    print_color("Your fork is now up to date with upstream.", "green")


def main():
    parser = argparse.ArgumentParser(description="Sync fork main branch with upstream")
    parser.add_argument(
        "--branch", default="main", help="Branch to sync (default: main)"
    )
    args = parser.parse_args()

    try:
        sync_fork(args.branch)
    except KeyboardInterrupt:
        print()
        print_color("Operation cancelled by user.", "yellow")
        sys.exit(0)


if __name__ == "__main__":
    main()
