"""
agent_clone_inspector.py — Clone Inspector specialist agent.

Responsibility: Detect whether the LLM-generated next generation files are
copied or semantically equivalent to the parent generation.

This goes beyond the spawner's byte-level identity check. It uses the LLM
to detect:
  - Byte-identical copies (trivial clone)
  - Near-identical copies (variable renames, comment changes only)
  - Structural clones (same algorithm, different syntax)
  - Prompt clones (same prompt template with trivial wording changes)
  - Genuine improvements (meaningfully different logic or strategy)

Forbidden from: Modifying any code. Reports only — never rewrites.

Information contract:
  Receives: Dict[filename → parent_source] + Dict[filename → child_source]
  Produces: OriginalityReport
"""
import difflib
import logging
from dataclasses import dataclass, field
from typing import Dict, List

from agent_base import SpecialistAgent
from llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class FileOriginality:
    """Originality assessment for a single file."""
    filename: str
    is_identical: bool          # byte-for-byte match
    similarity_pct: float       # 0–100, difflib ratio
    verdict: str                # "CLONE" | "NEAR_CLONE" | "IMPROVED" | "NEW"
    observations: str           # what specifically changed or didn't


@dataclass
class OriginalityReport:
    """Full originality assessment for an entire generation's file set."""
    passed: bool                              # True = genuinely new, False = clone detected
    verdict: str                              # "ORIGINAL" | "CLONE" | "NEAR_CLONE"
    originality_score: float                  # 0.0 (full clone) to 1.0 (fully original)
    identical_files: List[str] = field(default_factory=list)
    near_clone_files: List[str] = field(default_factory=list)   # >85% similar
    improved_files: List[str] = field(default_factory=list)
    file_reports: List[FileOriginality] = field(default_factory=list)
    llm_assessment: str = ""                  # LLM's overall judgment
    recommendation: str = ""                  # what to do next

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "verdict": self.verdict,
            "originality_score": round(self.originality_score, 3),
            "identical_files": self.identical_files,
            "near_clone_files": self.near_clone_files,
            "improved_files": self.improved_files,
            "llm_assessment": self.llm_assessment,
            "recommendation": self.recommendation,
            "file_reports": [
                {
                    "filename": r.filename,
                    "is_identical": r.is_identical,
                    "similarity_pct": round(r.similarity_pct, 1),
                    "verdict": r.verdict,
                    "observations": r.observations,
                }
                for r in self.file_reports
            ],
        }


# Files where similarity is expected/acceptable (data contracts, base classes)
# These must still be present but don't need radical changes
STABLE_FILES = {"contracts.py", "agent_base.py"}

# Threshold above which a file is considered a clone
CLONE_THRESHOLD = 0.95      # 95%+ similar → CLONE
NEAR_CLONE_THRESHOLD = 0.85  # 85–95% similar → NEAR_CLONE


class CloneInspectorAgent(SpecialistAgent):
    """
    Inspects LLM-generated offspring files for copying from the parent.

    Two-stage check:
      Stage 1 — Fast difflib check: flags byte-identical and near-identical files
      Stage 2 — LLM semantic check: sends the flagged files to the LLM to assess
                whether similarity is legitimate (e.g. contracts.py rarely changes)
                or whether the logic is a semantic clone.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__("clone_inspector", llm_client)

    def run(  # type: ignore[override]
        self,
        parent_files: Dict[str, str],
        child_files: Dict[str, str],
        required_files: List[str],
    ) -> OriginalityReport:
        self._log_start({
            "parent_files": f"{len(parent_files)} files",
            "child_files": f"{len(child_files)} files",
        })

        # Stage 1: Fast difflib scan
        file_reports = []
        for fname in required_files:
            parent_src = parent_files.get(fname, "").strip()
            child_src = child_files.get(fname, "").strip()

            if not parent_src:
                # No parent to compare against → treat as new
                file_reports.append(FileOriginality(
                    filename=fname,
                    is_identical=False,
                    similarity_pct=0.0,
                    verdict="NEW",
                    observations="No parent file to compare against.",
                ))
                continue

            if not child_src:
                file_reports.append(FileOriginality(
                    filename=fname,
                    is_identical=False,
                    similarity_pct=0.0,
                    verdict="MISSING",
                    observations="File missing from child generation.",
                ))
                continue

            is_identical = parent_src == child_src
            ratio = difflib.SequenceMatcher(None, parent_src, child_src).ratio()
            similarity_pct = ratio * 100

            if is_identical:
                verdict = "CLONE"
            elif similarity_pct >= CLONE_THRESHOLD * 100:
                verdict = "CLONE"
            elif similarity_pct >= NEAR_CLONE_THRESHOLD * 100:
                verdict = "NEAR_CLONE"
            elif similarity_pct >= 50:
                verdict = "IMPROVED"
            else:
                verdict = "REWRITTEN"

            file_reports.append(FileOriginality(
                filename=fname,
                is_identical=is_identical,
                similarity_pct=similarity_pct,
                verdict=verdict,
                observations=self._diff_summary(parent_src, child_src, fname),
            ))

        # Collect flags
        identical = [r.filename for r in file_reports if r.is_identical]
        near_clones = [
            r.filename for r in file_reports
            if not r.is_identical and r.verdict in ("CLONE", "NEAR_CLONE")
            and r.filename not in STABLE_FILES
        ]
        improved = [
            r.filename for r in file_reports
            if r.verdict in ("IMPROVED", "REWRITTEN")
        ]

        flagged = [r for r in file_reports
                   if r.verdict in ("CLONE", "NEAR_CLONE")
                   and r.filename not in STABLE_FILES]

        # Stage 2: LLM semantic assessment on flagged files
        llm_assessment = ""
        recommendation = ""
        if flagged:
            llm_assessment, recommendation = self._llm_semantic_check(
                flagged, parent_files, child_files
            )
        else:
            llm_assessment = "No files flagged for semantic review. All required logic files show meaningful changes."
            recommendation = "SPAWN_APPROVED"

        # Overall score: fraction of required files that are genuinely improved
        non_stable_required = [f for f in required_files if f not in STABLE_FILES]
        if non_stable_required:
            improved_count = sum(
                1 for r in file_reports
                if r.filename in non_stable_required
                and r.verdict in ("IMPROVED", "REWRITTEN", "NEW")
            )
            originality_score = improved_count / len(non_stable_required)
        else:
            originality_score = 1.0

        # Decide pass/fail
        # Fail if: any non-stable file is identical OR LLM says REJECT
        hard_clones = [f for f in identical if f not in STABLE_FILES]
        passed = (
            len(hard_clones) == 0
            and "REJECT" not in recommendation.upper()
        )

        if not passed:
            verdict_str = "CLONE"
        elif near_clones:
            verdict_str = "NEAR_CLONE"
        else:
            verdict_str = "ORIGINAL"

        report = OriginalityReport(
            passed=passed,
            verdict=verdict_str,
            originality_score=originality_score,
            identical_files=identical,
            near_clone_files=near_clones,
            improved_files=improved,
            file_reports=file_reports,
            llm_assessment=llm_assessment,
            recommendation=recommendation,
        )

        self._log_end(report)
        self.logger.info(
            f"[clone_inspector] verdict={verdict_str} "
            f"score={originality_score:.0%} "
            f"identical={identical} near_clones={near_clones}"
        )
        return report

    # ── Stage 2: LLM semantic check ───────────────────────────────────────────

    def _llm_semantic_check(
        self,
        flagged: List[FileOriginality],
        parent_files: Dict[str, str],
        child_files: Dict[str, str],
    ):
        """Ask the LLM whether the high-similarity files are semantic clones."""
        file_blocks = []
        for r in flagged[:6]:  # Cap at 6 files to keep prompt manageable
            parent_snippet = (parent_files.get(r.filename) or "")[:800]
            child_snippet = (child_files.get(r.filename) or "")[:800]
            file_blocks.append(
                f"FILE: {r.filename}  (difflib similarity: {r.similarity_pct:.1f}%)\n"
                f"--- PARENT (first 800 chars) ---\n{parent_snippet}\n"
                f"--- CHILD (first 800 chars) ---\n{child_snippet}\n"
            )

        prompt = f"""You are a code originality auditor for a self-replicating AI system.
Your job: determine whether the CHILD generation files below are genuine improvements
or semantic clones of the PARENT generation.

A semantic clone is code that:
  - Uses the same algorithm with renamed variables
  - Has the same logic with different comments or docstrings
  - Is structurally identical with only whitespace/formatting changes
  - Has the same prompts with trivially reworded sentences

A genuine improvement is code that:
  - Uses a different algorithm or strategy
  - Has meaningfully different prompts (new reasoning steps, different structure)
  - Adds new functionality or fixes a real bug
  - Changes the control flow or data structures

FILES TO ASSESS:
{chr(10).join(file_blocks)}

Produce exactly two sections:

ASSESSMENT:
(For each file, one sentence: is it a semantic clone or a genuine improvement? Why?)

RECOMMENDATION:
(One of: SPAWN_APPROVED / REJECT_CLONE — followed by one sentence explaining)"""

        response = self._call_llm(prompt)

        assessment = self._extract_section(response, "ASSESSMENT:").strip()
        recommendation = self._extract_section(response, "RECOMMENDATION:").strip()
        if not recommendation:
            recommendation = "SPAWN_APPROVED" if "APPROVED" in response.upper() else "REJECT_CLONE"

        return assessment, recommendation

    # ── Diff summary ──────────────────────────────────────────────────────────

    def _diff_summary(self, parent: str, child: str, fname: str) -> str:
        """Generate a short human-readable diff summary."""
        if parent == child:
            return "Byte-for-byte identical to parent — no changes made."

        parent_lines = parent.splitlines()
        child_lines = child.splitlines()
        diff = list(difflib.unified_diff(
            parent_lines, child_lines, lineterm="", n=0
        ))

        added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))

        if added == 0 and removed == 0:
            return "Whitespace/formatting changes only."
        return f"+{added} lines added, -{removed} lines removed across {len(child_lines)} total lines."
