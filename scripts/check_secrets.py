"""Fail safely when tracked files or reachable Git history contain likely secrets.

The report includes only a path, line number, and rule name. It never prints the
matched value. The checks are intentionally conservative and complement, rather
than replace, the secret scanning and push protection provided by the Git host.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MAX_FILE_SIZE = 5 * 1024 * 1024


@dataclass(frozen=True)
class SecretPattern:
    name: str
    expression: re.Pattern[str]
    history_expression: str


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    location: str
    rule: str


SECRET_PATTERNS = (
    SecretPattern(
        "HUGGING_FACE_TOKEN",
        re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        r"hf_[A-Za-z0-9]{20,}",
    ),
    SecretPattern(
        "GITHUB_TOKEN",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        r"(gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    ),
    SecretPattern(
        "OPENAI_API_KEY",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
        r"sk-(proj-)?[A-Za-z0-9_-]{20,}",
    ),
    SecretPattern(
        "GOOGLE_API_KEY",
        re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
        r"AIza[0-9A-Za-z_-]{30,}",
    ),
    SecretPattern(
        "AWS_ACCESS_KEY",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        r"AKIA[0-9A-Z]{16}",
    ),
    SecretPattern(
        "SLACK_TOKEN",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
        r"xox[baprs]-[A-Za-z0-9-]{20,}",
    ),
    SecretPattern(
        "STRIPE_LIVE_KEY",
        re.compile(r"\bsk_live_[A-Za-z0-9]{16,}\b"),
        r"sk_live_[A-Za-z0-9]{16,}",
    ),
    SecretPattern(
        "PRIVATE_KEY",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
        r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    ),
    SecretPattern(
        "CREDENTIALS_IN_URL",
        re.compile(r"https?://[^/\s:@]+:[^@\s]+@"),
        r"https?://[^/[:space:]:@]+:[^@[:space:]]+@",
    ),
)

SECRET_ASSIGNMENT = re.compile(
    r"^\s*"
    r"([A-Z_][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY)"
    r"[A-Z0-9_]*)"
    r"\s*[:=]\s*(.*?)\s*(?:#.*)?$",
)

CONFIG_SUFFIXES = {".env", ".ini", ".json", ".toml", ".yaml", ".yml"}
PLACEHOLDER_PREFIXES = (
    "${",
    "<",
    "changeme",
    "example",
    "not-set",
    "placeholder",
    "replace-me",
    "replace_me",
    "your-",
    "your_",
    "xxx",
)


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def repository_files(root: Path, *, include_untracked: bool = False) -> Iterable[Path]:
    command = ["git", "ls-files", "-z"]
    if include_untracked:
        command.extend(["--cached", "--others", "--exclude-standard"])
    result = subprocess.run(
        command,
        cwd=root,
        check=True,
        capture_output=True,
    )
    for relative_path in result.stdout.decode("utf-8").split("\0"):
        if relative_path:
            yield root / relative_path


def is_configuration_file(path: Path) -> bool:
    return path.name.startswith(".env") or path.suffix.lower() in CONFIG_SUFFIXES


def is_placeholder(value: str) -> bool:
    normalized = value.strip().strip("\"'").strip().lower()
    return not normalized or normalized.startswith(PLACEHOLDER_PREFIXES)


def scan_text(path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    display_path = path.as_posix()
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern in SECRET_PATTERNS:
            if pattern.expression.search(line):
                findings.append(Finding(display_path, str(line_number), pattern.name))

        if is_configuration_file(path):
            assignment = SECRET_ASSIGNMENT.match(line)
            if assignment and not is_placeholder(assignment.group(2)):
                findings.append(
                    Finding(display_path, str(line_number), "NON_PLACEHOLDER_SECRET_ASSIGNMENT")
                )
    return findings


def scan_tree(root: Path, *, include_untracked: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    for path in repository_files(root, include_untracked=include_untracked):
        if not path.is_file() or path.stat().st_size > MAX_FILE_SIZE:
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        relative_path = path.relative_to(root)
        findings.extend(scan_text(relative_path, data.decode("utf-8", errors="replace")))
    return findings


def scan_history(root: Path) -> list[Finding]:
    findings: set[Finding] = set()
    commit_pattern = re.compile(r"^[0-9a-f]{40}$")
    for pattern in SECRET_PATTERNS:
        result = subprocess.run(
            [
                "git",
                "log",
                "--all",
                "--extended-regexp",
                "--format=%H",
                "--name-only",
                f"-G{pattern.history_expression}",
                "--",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = "unknown"
        for line in result.stdout.splitlines():
            if commit_pattern.fullmatch(line):
                commit = line[:12]
            elif line:
                findings.add(Finding(line, f"commit {commit}", pattern.name))
    return sorted(findings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan tracked files for likely credentials without printing their values."
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="also scan every commit reachable from local branches and tags",
    )
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="also scan untracked files that are not excluded by .gitignore",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repository_root()
    findings = scan_tree(root, include_untracked=args.include_untracked)
    if args.history:
        findings.extend(scan_history(root))

    findings = sorted(set(findings))
    if findings:
        print("Potential secrets found. Matched values are intentionally redacted:")
        for finding in findings:
            print(f"{finding.path}:{finding.location}: {finding.rule}")
        return 1

    scopes = ["tracked files"]
    if args.include_untracked:
        scopes.append("untracked non-ignored files")
    if args.history:
        scopes.append("reachable Git history")
    scope = ", ".join(scopes)
    print(f"Secret scan passed: no likely credentials found in {scope}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
