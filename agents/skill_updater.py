"""
Reads all SkillRunEntry records and rewrites SKILL.md with learned rules.
This is the self-improvement step — run after accumulating experience.
"""
import difflib
from collections import defaultdict
from pathlib import Path

from agents.skill_runner import load_skill_runs

SKILL_FILE = Path(__file__).parent.parent / "skills" / "memorycast" / "SKILL.md"

_PROFILE_LABELS = {
    "high_vol_weak_seas":   "High volatility + weak seasonality",
    "low_vol_strong_seas":  "Low volatility + strong seasonality",
    "high_vol_strong_seas": "High volatility + strong seasonality",
    "mixed":                "Mixed / unclear signals",
}
_PROFILE_THRESHOLDS = {
    "high_vol_weak_seas":   "volatility > 0.15 AND seasonality < 0.3",
    "low_vol_strong_seas":  "volatility ≤ 0.15 AND seasonality ≥ 0.4",
    "high_vol_strong_seas": "volatility > 0.15 AND seasonality ≥ 0.4",
    "mixed":                "all other profiles",
}


def _classify(profile: dict) -> str:
    vol = profile.get("volatility", 0)
    seas = profile.get("seasonality_strength", 0)
    if vol > 0.15 and seas < 0.3:
        return "high_vol_weak_seas"
    if vol <= 0.15 and seas >= 0.4:
        return "low_vol_strong_seas"
    if vol > 0.15 and seas >= 0.4:
        return "high_vol_strong_seas"
    return "mixed"


def _best_model(runs: list[dict]) -> tuple[str, float, list[str]]:
    scores: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        scores[r["strategy_used"]].append(r["success_score"])
    best = max(scores, key=lambda m: sum(scores[m]) / len(scores[m]))
    avg = round(sum(scores[best]) / len(scores[best]), 3)
    evidence = [
        f"{r['series_label']} — {r['strategy_used']} won, WAPE {r['winner_wape']:.1%}, score {r['success_score']:.2f}"
        for r in sorted(runs, key=lambda x: x["winner_wape"])[:3]
    ]
    return best, avg, evidence


def _generate_v2(runs: list[dict]) -> str:
    groups: dict[str, list] = defaultdict(list)
    for r in runs:
        groups[_classify(r["series_profile"])].append(r)

    avg_imp = round(sum(r["improvement"] for r in runs) / len(runs) * 100, 1)
    best_run = min(runs, key=lambda r: r["winner_wape"])

    rule_blocks = []
    for key, label in _PROFILE_LABELS.items():
        group = groups.get(key, [])
        if not group:
            continue
        best, avg_score, evidence = _best_model(group)
        ev_lines = "\n".join(f"    - {e}" for e in evidence)
        rule_blocks.append(
            f"- **{label}** ({_PROFILE_THRESHOLDS[key]})\n"
            f"  → Preferred model: **{best}** (avg success score: {avg_score:.2f} across {len(group)} run(s))\n"
            f"  → Evidence:\n{ev_lines}"
        )

    rules_text = "\n\n".join(rule_blocks) if rule_blocks else "_Not enough data yet._"
    summary = (
        f"Total runs: {len(runs)} | "
        f"Avg improvement over baseline: {avg_imp}pp | "
        f"Best run: {best_run['series_label']} ({best_run['strategy_used']}, {best_run['winner_wape']:.1%} WAPE)"
    )

    return f"""# Forecasting Strategy Skill — v2
> Auto-updated from {len(runs)} SkillRunEntry records stored in Cognee memory.

## Core Strategy
1. Profile the series (volatility, seasonality, trend strength)
2. Retrieve similar SkillRunEntry records from Cognee memory
3. Apply learned rules below — skip to full tournament if signals are mixed
4. Run cross-validation tournament (3 windows) to confirm
5. Record new SkillRunEntry with success_score and feedback
6. Re-run skill_updater to incorporate new evidence into next version

## Learned Rules (from run history)
{rules_text}

## Fallback Rule
When fewer than 2 similar past runs exist, run the full tournament:
SeasonalNaive → AutoARIMA → AutoETS → GradientBoosting
Select by lowest cross-validation WAPE.

## Feature Engineering (GradientBoosting)
- Lag features: 1, 2, 3, 6, 12 months back
- Date features: month of year
- Evidence suggests lag-12 is critical for annual patterns

## Run History Summary
{summary}

## Cognee Integration
Each run is stored as a SkillRunEntry with:
- `success_score` = 1 - winner_wape
- `improvement` = baseline_wape - winner_wape
- `feedback` = natural language explanation
- `strategy_used` = winning model name

## What Changed from v1
- Added learned rules per profile type (derived from real run evidence)
- Added evidence citations for each rule
- Added run history summary
- Fallback rule now explicit
"""


def update_skill() -> tuple[str, str, str]:
    """
    Generate v2 from run history.
    Returns (v1_text, v2_text, unified_diff).
    Does NOT overwrite SKILL.md — diff is shown in the UI.
    """
    v1_text = SKILL_FILE.read_text()
    runs = load_skill_runs()

    if not runs:
        return v1_text, v1_text, "No SkillRunEntry records found — run forecasts first."

    v2_text = _generate_v2(runs)

    diff_lines = list(difflib.unified_diff(
        v1_text.splitlines(keepends=True),
        v2_text.splitlines(keepends=True),
        fromfile="SKILL.md  (v1 — baseline)",
        tofile="SKILL.md  (v2 — learned)",
        lineterm="\n",
    ))
    diff_text = "".join(diff_lines)

    return v1_text, v2_text, diff_text


if __name__ == "__main__":
    v1, v2, diff = update_skill()
    print(diff)
