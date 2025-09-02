#!/usr/bin/env python3
"""
Temporary verifier for README.md

Runs executable code blocks and checks selected claims.

Notes:
- Python snippets are executed in-process with PYTHONPATH set to project root.
- Bash snippets are skipped by default for safety; known local verify scripts are run separately.
- JSON snippets are validated for syntax.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"


@dataclass
class SnippetResult:
    index: int
    language: str
    summary: str
    status: str  # PASS/FAIL/SKIPPED
    error: Optional[str] = None


def read_readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def extract_code_blocks(md: str) -> List[tuple[str, str]]:
    blocks: List[tuple[str, str]] = []
    # Match triple backtick blocks with optional language
    pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)\n```", re.DOTALL)
    for match in pattern.finditer(md):
        lang = match.group(1).strip().lower()
        code = match.group(2)
        blocks.append((lang, code))
    return blocks


def run_python_snippet(code: str) -> tuple[bool, Optional[str]]:
    # Heuristic skips for clearly non-executable illustrative snippets
    non_exec_markers = [
        "skip_connection_rule",
        "lstm_insertion_rule",
        "layer_widening_rule",
    ]
    if any(m in code for m in non_exec_markers):
        return False, "non-executable example (uses placeholder rules)"

    # Prepare an isolated globals dict with project on sys.path
    env_globals = {"__name__": "__main__"}
    env_locals: dict = {}
    sys_path_added = False
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
            sys_path_added = True
        prelude = (
            "from ggnes.core import Graph, NodeType\n"
            "from ggnes.translation import to_pytorch_model\n"
            "from ggnes.hierarchical.module_spec import ModuleSpec\n"
        )
        exec(prelude + "\n" + code, env_globals, env_locals)
        return True, None
    except Exception as exc:  # noqa: BLE001 - we need to capture any failure
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if sys_path_added:
            try:
                sys.path.remove(str(REPO_ROOT))
            except ValueError:
                pass


def validate_json_snippet(code: str) -> tuple[bool, Optional[str]]:
    try:
        json.loads(code)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    md = read_readme()
    blocks = extract_code_blocks(md)
    results: List[SnippetResult] = []

    for idx, (lang, code) in enumerate(blocks, start=1):
        if lang in {"mermaid", "", "text"}:
            results.append(SnippetResult(idx, lang or "plain", "diagram/text", "SKIPPED"))
            continue
        if lang == "python":
            ok, err = run_python_snippet(code)
            status = "PASS" if ok else ("SKIPPED" if err and "non-executable" in err else "FAIL")
            results.append(SnippetResult(idx, lang, "python snippet", status, err))
            continue
        if lang in {"bash", "sh"}:
            # For safety, do not execute arbitrary bash from README.
            results.append(SnippetResult(idx, lang, "bash snippet (not executed)", "SKIPPED"))
            continue
        if lang == "json":
            ok, err = validate_json_snippet(code)
            results.append(SnippetResult(idx, lang, "json snippet", "PASS" if ok else "FAIL", err))
            continue
        # Unknown languages are skipped
        results.append(SnippetResult(idx, lang, "unknown language", "SKIPPED"))

    # Run known local verification scripts mentioned in README
    bundle_checks: List[SnippetResult] = []
    for script_rel in [
        "repro_bundles/mnist_demo/verify.sh",
        "repro_bundles/binary_classification/verify.sh",
    ]:
        script_path = REPO_ROOT / script_rel
        if not script_path.exists():
            bundle_checks.append(
                SnippetResult(-1, "bash", f"{script_rel}", "SKIPPED", "missing script")
            )
            continue
        try:
            # Run without args; ensure venv python is on PATH so "python" resolves
            venv_bin = Path(sys.executable).parent
            env = os.environ.copy()
            env_path = os.pathsep.join([str(venv_bin), env.get("PATH", "")])
            env.update({"PATH": env_path, "PYTHONPATH": str(REPO_ROOT)})
            subprocess.run(
                ["bash", str(script_path)],
                cwd=script_path.parent,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            bundle_checks.append(SnippetResult(-1, "bash", f"{script_rel}", "PASS"))
        except subprocess.CalledProcessError as exc:
            bundle_checks.append(SnippetResult(-1, "bash", f"{script_rel}", "FAIL", exc.stderr.decode("utf-8", errors="ignore") or str(exc)))

    # Optionally run tests to verify the passing tests claim
    tests_status: Optional[SnippetResult] = None
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        out = proc.stdout.decode("utf-8", errors="ignore")
        ok = proc.returncode == 0
        summary_line = next((line for line in out.splitlines() if re.search(r"\bpassed\b", line)), "").strip()
        tests_status = SnippetResult(-1, "pytest", "test suite", "PASS" if ok else "FAIL", summary_line or out[-500:])
    except FileNotFoundError:
        tests_status = SnippetResult(-1, "pytest", "test suite", "SKIPPED", "pytest not available")

    # Print report
    print("=== README Snippet Verification ===")
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIPPED")
    print(f"Snippets: PASS={passed} FAIL={failed} SKIPPED={skipped} (total={len(results)})")
    for r in results:
        msg = f"[{r.status}] #{r.index} {r.language}: {r.summary}"
        if r.error and r.status != "PASS":
            msg += f" — {r.error}"
        print(msg)

    print("\n=== Bundle Verification Scripts ===")
    for r in bundle_checks:
        msg = f"[{r.status}] {r.summary}"
        if r.error and r.status != "PASS":
            msg += f" — {r.error.strip()}"
        print(msg)

    print("\n=== Test Suite Status ===")
    if tests_status is not None:
        msg = f"[{tests_status.status}] {tests_status.summary}"
        if tests_status.error:
            msg += f" — {tests_status.error}"
        print(msg)
    else:
        print("[SKIPPED] tests not executed")

    # Exit non-zero if any snippet or bundle check failed
    any_fail = failed > 0 or any(r.status == "FAIL" for r in bundle_checks) or (tests_status and tests_status.status == "FAIL")
    return 1 if any_fail else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))
    sys.exit(main())


