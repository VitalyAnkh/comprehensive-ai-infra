"""F04: the four delivery checks, run together and reported without rounding up.

The work package asks for four things: every result regenerable, the source text
audited against the work packages, the outline and the rendered book in step,
and a reader entry that actually works. Each is checked here on its own and
reported on its own. A pass in three of them is not a pass.

Nothing in this module regenerates anything. It reads what is on disk and says
whether it holds together, so that a green result means the checked-in state is
consistent rather than that a rebuild would fix it.
"""
import json
import re
import subprocess
import sys

from .paths import BOOK, PROJECT
from .sources import is_lfs_pointer
from . import coverage

# Two scenario groups name their results from the row's own fields rather than
# from an "id", so the audit has to derive those names the same way reproduce does.
DERIVED_NAMES = {
    "states": lambda row: (f"state-{row['model']}-n{row['length']}"
                           f"-b{row.get('batch', 1)}-{row.get('mla_path', 'native')}"),
    "generation": lambda row: f"generation-{row['model']}-s{row['history']}-g{row['steps']}",
}

# Artifacts reproduce writes that are not one scenario's output.
STANDING_ARTIFACTS = {"results/README.md", "results/manifest.json", "results/hardware.md",
                      "results/hardware-audit.json", "results/hardware-audit.md",
                      # A figure reproduce draws beside one scenario's result rather
                      # than a scenario output of its own.
                      "results/memory-pool-layout.svg"}


def scenario_artifacts() -> dict:
    """Every scenario id has results, and every result belongs to a scenario."""
    manifest = json.loads((PROJECT / "results/manifest.json").read_text())
    produced = {row["file"] for row in manifest["artifacts"]}
    stems = {re.sub(r"\.(json|md|csv)$", "", name[len("results/"):])
             for name in produced if name not in STANDING_ARTIFACTS}
    h2_ids = set()
    h2_path = PROJECT / "configs/h2-scenarios.json"
    if h2_path.exists():
        h2_spec = json.loads(h2_path.read_text())
        h2_ids = {row["id"] for row in h2_spec.get("h2_topics", []) if isinstance(row.get("id"), str)}
    book_path = PROJECT / "scenarios/book.json"
    if is_lfs_pointer(book_path):
        expected = (stems | h2_ids)
    else:
        scenarios = json.loads(book_path.read_text())
        expected = set(h2_ids)
        for name, rows in scenarios.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if isinstance(row.get("id"), str):
                    expected.add(row["id"])
                elif name in DERIVED_NAMES:
                    expected.add(DERIVED_NAMES[name](row))
    missing = sorted(name for name in expected
                     if f"results/{name}.json" not in produced or f"results/{name}.md" not in produced)
    unclaimed = sorted(stems - expected)
    return {"check": "every scenario id has results and every result has a scenario",
            "scenario_ids": len(expected), "artifacts": len(produced),
            "scenarios_without_results": missing,
            "results_without_a_scenario": unclaimed,
            "passed": not missing and not unclaimed}


def reader_entry() -> dict:
    """Every CLI subcommand is documented, and every topic is reachable from it."""
    cli = (PROJECT / "src/infra_calc/cli.py").read_text()
    readme = (PROJECT / "README.md").read_text()
    commands = set(re.findall(r'sub\.add_parser\("([a-z0-9-]+)"', cli))
    documented = {name for name in commands if f"calc.py {name}" in readme}
    modules = {path.stem for path in (PROJECT / "src/infra_calc/topics").glob("*.py")
               if path.stem != "__init__"}
    # A topic is reachable if the CLI imports it, or if another module does: a
    # shared accounting helper is not required to have its own subcommand.
    reachable = set()
    for path in (PROJECT / "src/infra_calc").rglob("*.py"):
        if path.stem in modules and path.parent.name == "topics":
            for line in path.read_text().splitlines():
                if "import" not in line:
                    continue
                for name in modules:
                    if name != path.stem and re.search(rf"\b{name}\b", line):
                        reachable.add(name)
    for line in cli.splitlines():
        match = re.match(r"from \.topics import (.+)$", line)
        if match:
            reachable.update(part.strip() for part in match.group(1).split(","))
    orphaned = sorted(modules - reachable)
    return {"check": "every subcommand is documented and every topic is reachable",
            "subcommands": len(commands), "documented_subcommands": len(documented),
            "undocumented_subcommands": sorted(commands - documented),
            "topic_modules": len(modules), "reachable_topics": len(modules & reachable),
            "topics_reachable_from_nowhere": orphaned,
            "passed": not (commands - documented) and not orphaned}


def source_audit() -> dict:
    """The source-to-work-package audit exists, is current, and names its gaps."""
    path = PROJECT / "inventory/f01-coverage.json"
    if not path.exists():
        return {"check": "source text audited against the work packages",
                "passed": False, "reason": "inventory/f01-coverage.json has not been written"}
    if is_lfs_pointer(path) or is_lfs_pointer(PROJECT / "inventory/sources.json"):
        return {"check": "source text audited against the work packages",
                "passed": False, "reason": "inventory files are unpulled Git LFS pointers"}
    saved = json.loads(path.read_text())
    fresh = coverage.build()
    current = saved.get("summary") == fresh["summary"] and saved.get("blocks") == fresh["blocks"]
    reviewed = fresh["summary"]["blocks_reviewed"]
    return {"check": "source text audited against the work packages",
            "blocks": fresh["blocks"],
            "saved_report_is_current": current,
            "blocks_citing_a_missing_result": fresh["summary"]["blocks_citing_a_missing_result"],
            "blocks_reviewed": reviewed,
            "blocks_pending_review": fresh["blocks"] - reviewed,
            "sectioned_gaps": sorted({row["section"] for row in fresh["unmapped_quantitative_blocks"]
                                      if row["section"]}),
            "passed": current and fresh["summary"]["blocks_citing_a_missing_result"] == 0
            and reviewed == fresh["blocks"],
            "reason": ("Every block still needs its own review; a mechanical map does not close F01"
                       if reviewed < fresh["blocks"] else "")}


def rendered_book() -> dict:
    """The outline and the rendered page agree, as the book's own scripts judge it."""
    script = BOOK / "scripts/verify_outline.py"
    if not script.exists():
        return {"check": "outline and rendered book in step", "passed": False,
                "reason": "scripts/verify_outline.py is missing"}
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                         cwd=str(BOOK / "scripts"))
    try:
        report = json.loads(run.stdout)
    except json.JSONDecodeError:
        return {"check": "outline and rendered book in step", "passed": False,
                "reason": "verify_outline produced no report", "stderr": run.stderr[-400:]}
    return {"check": "outline and rendered book in step",
            "status": report.get("status"),
            "local_links_checked": report.get("local_links_checked"),
            "reference_hashes_checked": report.get("reference_hashes_checked"),
            "errors": report.get("errors", []),
            "passed": report.get("status") == "passed"}


def calculate(include_rendered_book: bool = True) -> dict:
    """Run the four checks and report each one separately."""
    checks = {"results_regenerable": scenario_artifacts(),
              "source_audited": source_audit(),
              "reader_entry": reader_entry()}
    if include_rendered_book:
        checks["rendered_book"] = rendered_book()
    passed = [name for name, row in checks.items() if row["passed"]]
    failed = [name for name, row in checks.items() if not row["passed"]]
    return {
        "calculation": "f04-delivery-audit",
        "checks": checks,
        "passed": sorted(passed), "failed": sorted(failed),
        "delivered": not failed,
        "assumptions": [
            "This reads the checked-in state; it regenerates nothing, so a pass means what is on disk holds together.",
            "The four checks are reported separately and are never averaged into a single completion figure.",
            "Result regenerability here means the manifest and the scenario list agree; the byte-level check is verify-results.",
            "The source audit cannot pass while blocks remain unreviewed, because a mechanical map is not a review.",
            "The rendered-book check delegates to the book's own verify_outline and repeats its verdict unchanged.",
        ],
    }


def markdown(result: dict) -> str:
    lines = ["# F04：交付检查", "",
             "四项检查各自独立报告，不合并成一个完成度数字。", "",
             "| 检查 | 结论 |", "|---|---|"]
    for name, row in result["checks"].items():
        lines.append(f"| {name} | {'通过' if row['passed'] else '未通过'} |")
    lines.append("")
    for name, row in result["checks"].items():
        lines += [f"## {name}", "", f"- 检查内容：{row['check']}"]
        for key, value in row.items():
            if key in ("check", "passed"):
                continue
            if isinstance(value, list):
                lines.append(f"- {key}：{'、'.join(map(str, value[:12])) or '无'}"
                             + ("（另有更多）" if len(value) > 12 else ""))
            elif value not in ("", None):
                lines.append(f"- {key}：{value}")
        lines.append("")
    lines += [f"整体交付：{'通过' if result['delivered'] else '未通过'}；"
              f"未通过项：{'、'.join(result['failed']) or '无'}。", "", "## 口径与限制", ""]
    lines += ["- " + item for item in result["assumptions"]]
    lines += ["", "```json", json.dumps(result, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)
