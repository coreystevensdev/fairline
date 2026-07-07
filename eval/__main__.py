"""Steambot eval harness.

Tests three layers of deterministic math against golden fixtures, then checks
structural validity of pick output. No network calls, no API keys required.

Usage:
    python -m eval
    python -m eval --out eval/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fairline.state import american_to_prob, remove_vig

DATASET = Path(__file__).parent / "dataset.jsonl"


# Math under test -- mirrors pick.py _compute_ev and state.py functions


def compute_ev(blended_prob: float, american_price: int) -> float:
    if american_price > 0:
        win_amount = american_price / 100
    else:
        win_amount = 100 / abs(american_price)
    return blended_prob * win_amount - (1 - blended_prob) * 1.0


def compute_clv(blended_probability: float, closing_probability: float) -> float:
    return closing_probability - blended_probability


def compute_edge(blended_prob: float, american_price: int) -> float:
    implied = american_to_prob(american_price)
    return blended_prob - implied


# Result types


@dataclass
class CaseResult:
    case_id: str
    category: str
    description: str
    passed: bool
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)

    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    def failed(self) -> int:
        return sum(1 for c in self.cases if not c.passed)

    def total(self) -> int:
        return len(self.cases)

    def pass_rate(self) -> float:
        return self.passed() / self.total() if self.total() else 0.0

    def by_category(self) -> dict[str, dict[str, int]]:
        categories: dict[str, dict[str, int]] = {}
        for c in self.cases:
            bucket = categories.setdefault(c.category, {"passed": 0, "failed": 0})
            if c.passed:
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1
        return categories


# Case runners


def _near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def run_vig_removal(case: dict) -> CaseResult:
    inputs = case["inputs"]
    expected = case["expected"]
    prices: list[int] = inputs["prices"]
    tol: float = expected["tolerance"]

    raw = [american_to_prob(p) for p in prices]
    fair = remove_vig(raw)

    if "ratio_preserved" in expected:
        ratio_raw = raw[0] / raw[1]
        ratio_fair = fair[0] / fair[1]
        passed = _near(ratio_raw, ratio_fair, tol)
        return CaseResult(
            case_id=case["id"],
            category=case["category"],
            description=case["description"],
            passed=passed,
            details={"ratio_raw": ratio_raw, "ratio_fair": ratio_fair},
        )

    expected_probs: list[float] = expected["fair_probs"]
    errors = [abs(fair[i] - expected_probs[i]) for i in range(len(fair))]
    passed = all(e <= tol for e in errors) and _near(sum(fair), 1.0, 1e-9)
    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        description=case["description"],
        passed=passed,
        details={"computed": fair, "expected": expected_probs, "errors": errors, "sum": sum(fair)},
    )


def run_ev_calculation(case: dict) -> CaseResult:
    inputs = case["inputs"]
    expected = case["expected"]
    tol = expected["tolerance"]

    ev = compute_ev(inputs["blended_prob"], inputs["american_price"])

    if "ev_near_zero" in expected:
        passed = _near(ev, 0.0, tol)
    else:
        passed = _near(ev, expected["ev"], tol)

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        description=case["description"],
        passed=passed,
        details={"computed_ev": ev, "expected_ev": expected.get("ev", "~0")},
    )


def run_clv_calculation(case: dict) -> CaseResult:
    inputs = case["inputs"]
    expected = case["expected"]
    clv = compute_clv(inputs["blended_probability"], inputs["closing_probability"])
    passed = _near(clv, expected["clv"], expected["tolerance"])
    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        description=case["description"],
        passed=passed,
        details={"computed_clv": clv, "expected_clv": expected["clv"]},
    )


def run_edge_filter(case: dict) -> CaseResult:
    inputs = case["inputs"]
    expected = case["expected"]
    tol = expected["tolerance"]

    edge = compute_edge(inputs["blended_prob"], inputs["american_price"])
    min_edge = inputs["min_edge_pct"]
    passes_filter = edge >= min_edge

    edge_ok = _near(edge, expected["edge_pct"], tol) if "edge_pct" in expected else True
    filter_ok = passes_filter == expected["passes_filter"]

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        description=case["description"],
        passed=edge_ok and filter_ok,
        details={
            "computed_edge": edge,
            "expected_edge": expected.get("edge_pct"),
            "passes_filter": passes_filter,
            "expected_filter": expected["passes_filter"],
        },
    )


VALID_CONFIDENCE = {"high", "medium", "low"}
REQUIRED_PICK_FIELDS = {
    "pick_id", "game_id", "home_team", "away_team", "commence_time",
    "market", "selection", "best_book", "best_price", "sharp_probability",
    "blended_probability", "implied_probability", "edge_pct", "ev_pct",
    "confidence", "rationale",
}


def run_structural(case: dict) -> CaseResult:
    inputs = case["inputs"]
    expected = case["expected"]

    if "pick" in inputs:
        pick = inputs["pick"]

        if not expected.get("valid", True):
            # Invalid case: check that we can detect the violation
            reason = expected.get("reason", "")
            if "confidence" in reason and "confidence" in pick:
                caught = pick["confidence"] not in VALID_CONFIDENCE
            else:
                caught = False
            return CaseResult(
                case_id=case["id"],
                category=case["category"],
                description=case["description"],
                passed=caught,
                details={"reason": reason, "caught": caught},
            )

        missing = REQUIRED_PICK_FIELDS - set(pick.keys())
        bad_confidence = pick.get("confidence") not in VALID_CONFIDENCE
        negative_edge = pick.get("edge_pct", 0) < 0

        passed = not missing and not bad_confidence and not negative_edge
        return CaseResult(
            case_id=case["id"],
            category=case["category"],
            description=case["description"],
            passed=passed,
            details={
                "missing_fields": list(missing),
                "bad_confidence": bad_confidence,
                "negative_edge": negative_edge,
            },
        )

    if "blended_probability" in inputs:
        # Consistency check: edge_pct == blended - implied (within tolerance)
        tol = expected["tolerance"]
        computed_implied = american_to_prob(-110)  # placeholder; use direct values
        computed_edge = inputs["blended_probability"] - inputs["implied_probability"]
        passed = _near(computed_edge, inputs["edge_pct"], tol)
        return CaseResult(
            case_id=case["id"],
            category=case["category"],
            description=case["description"],
            passed=passed,
            details={"computed_edge": computed_edge, "stated_edge": inputs["edge_pct"]},
        )

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        description=case["description"],
        passed=False,
        error="Unknown structural case shape",
    )


RUNNERS = {
    "vig_removal": run_vig_removal,
    "ev_calculation": run_ev_calculation,
    "clv_calculation": run_clv_calculation,
    "edge_filter": run_edge_filter,
    "structural": run_structural,
}


# Main


def load_dataset(path: Path) -> list[dict]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_eval(dataset_path: Path = DATASET) -> EvalReport:
    cases = load_dataset(dataset_path)
    report = EvalReport()

    for case in cases:
        category = case.get("category", "")
        runner = RUNNERS.get(category)
        if runner is None:
            report.cases.append(
                CaseResult(
                    case_id=case["id"],
                    category=category,
                    description=case.get("description", ""),
                    passed=False,
                    error=f"No runner for category '{category}'",
                )
            )
            continue

        try:
            result = runner(case)
        except Exception as exc:
            result = CaseResult(
                case_id=case["id"],
                category=category,
                description=case.get("description", ""),
                passed=False,
                error=str(exc),
            )
        report.cases.append(result)

    return report


def _print_report(report: EvalReport) -> None:
    col_width = 10
    pass_sym = "PASS"
    fail_sym = "FAIL"

    print()
    print(f"{'Case':<14} {'Category':<20} {'Result':<6}  Description")
    print("-" * 90)
    for c in report.cases:
        sym = pass_sym if c.passed else fail_sym
        print(f"{c.case_id:<14} {c.category:<20} {sym:<6}  {c.description}")
        if not c.passed:
            if c.error:
                print(f"{'':>14}   error: {c.error}")
            elif c.details:
                print(f"{'':>14}   details: {json.dumps(c.details)}")

    print()
    print("By category:")
    for cat, counts in report.by_category().items():
        p = counts["passed"]
        f = counts["failed"]
        t = p + f
        bar = "#" * p + "." * f
        print(f"  {cat:<20} {p}/{t}  [{bar}]")

    print()
    total = report.total()
    passed = report.passed()
    failed = report.failed()
    print(f"Total: {passed}/{total} passed  ({failed} failed)  pass rate: {report.pass_rate():.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Steambot eval harness")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path, help="Write JSON report to this path")
    args = parser.parse_args()

    report = run_eval(args.dataset)
    _print_report(report)

    if args.out:
        payload = {
            "total": report.total(),
            "passed": report.passed(),
            "failed": report.failed(),
            "pass_rate": report.pass_rate(),
            "by_category": report.by_category(),
            "cases": [
                {
                    "id": c.case_id,
                    "category": c.category,
                    "description": c.description,
                    "passed": c.passed,
                    "error": c.error,
                    "details": c.details,
                }
                for c in report.cases
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nReport written to {args.out}")

    return 0 if report.failed() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
