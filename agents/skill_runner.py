"""
Logs a SkillRunEntry after each forecasting tournament.
JSON primary; Cognee remember queued on the shared Cognee worker loop.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

from agents.memory import _fire, _cognee_setup, _remember

SKILL_RUNS_FILE = Path(__file__).parent.parent / "memory" / "skill_runs.json"


def _format_run_text(entry: dict) -> str:
    sp = entry.get("series_profile", {})
    return (
        f"SkillRunEntry forecasting run. "
        f"Series: {entry['series_label']}. "
        f"Strategy: {entry['strategy_used']} won with WAPE={entry['winner_wape']:.1%}, "
        f"{entry['improvement']*100:.1f}pp better than SeasonalNaive baseline. "
        f"Success score: {entry['success_score']:.2f}. "
        f"Profile: volatility={sp.get('volatility', 0):.2f}, "
        f"seasonality={sp.get('seasonality_strength', 0):.2f}, "
        f"trend={sp.get('trend_strength', 0):.2f}. "
        f"Feedback: {entry['feedback']}."
    )


def _build_feedback(
    series_label: str,
    profile: dict,
    results: dict,
    winner: str,
    baseline_wape: float,
    winner_wape: float,
    improvement: float,
) -> str:
    vol = profile["volatility"]
    seas = profile["seasonality_strength"]
    trend = profile["trend_strength"]
    vol_label = "high-volatility" if vol > 0.15 else "low-volatility"
    seas_label = "strong-seasonality" if seas > 0.4 else "weak-seasonality"
    trend_label = "strong-trend" if trend > 0.5 else "weak-trend"

    # Rank all models by WAPE
    ranked = sorted(results.items(), key=lambda x: x[1]["wape"])
    tournament_str = ", ".join(f"{m}={v['wape']:.1%}" for m, v in ranked)

    # Runner-up for context
    losers = [(m, v["wape"]) for m, v in ranked if m != winner]
    runner_up = f"vs runner-up {losers[0][0]} ({losers[0][1]:.1%})" if losers else ""

    # Why the winner likely won
    if winner == "AutoARIMA":
        why = "handles autocorrelated residuals and level shifts well"
    elif winner == "AutoETS":
        why = "captures exponential smoothing patterns in seasonal/trend structure"
    elif winner == "GradientBoosting":
        why = "exploits lag features to model nonlinear volatility patterns"
    elif winner == "SeasonalNaive":
        why = "strong seasonal pattern made simple repetition competitive"
    else:
        why = "best fit for this series profile"

    return (
        f"{winner} won on '{series_label}' ({vol_label}, {seas_label}, {trend_label}). "
        f"Profile: CV={vol:.2f}, seasonality={seas:.2f}, trend R²={trend:.2f}. "
        f"Tournament: {tournament_str}. "
        f"{winner} achieved {winner_wape:.1%} WAPE — "
        f"{improvement * 100:.1f}pp better than SeasonalNaive baseline ({baseline_wape:.1%}) {runner_up}. "
        f"Why: {why}."
    )


def log_skill_run(
    series_label: str,
    profile: dict,
    results: dict,
    winner: str,
    lesson_text: str,
) -> dict:
    """Create and persist a SkillRunEntry from a tournament result."""
    baseline_wape = results.get("SeasonalNaive", {}).get("wape", 1.0)
    winner_wape = results[winner]["wape"]
    improvement = round(baseline_wape - winner_wape, 4)
    success_score = round(max(0.0, 1.0 - winner_wape), 4)

    entry = {
        "run_id": str(uuid.uuid4())[:8],
        "series_label": series_label,
        "series_profile": {
            "volatility": profile["volatility"],
            "seasonality_strength": profile["seasonality_strength"],
            "trend_strength": profile["trend_strength"],
        },
        "strategy_used": winner,
        "success_score": success_score,
        "baseline_wape": round(baseline_wape, 4),
        "winner_wape": winner_wape,
        "improvement": improvement,
        "feedback": _build_feedback(
            series_label, profile, results, winner, baseline_wape, winner_wape, improvement
        ),
        "lesson_text": lesson_text,
        "timestamp": datetime.utcnow().isoformat(),
    }

    _persist(entry)
    _cognee_setup()
    _fire(_remember(_format_run_text(entry)))

    return entry


def load_skill_runs() -> list[dict]:
    if not SKILL_RUNS_FILE.exists():
        return []
    return json.loads(SKILL_RUNS_FILE.read_text())


def _persist(entry: dict):
    SKILL_RUNS_FILE.parent.mkdir(exist_ok=True)
    runs = load_skill_runs()
    runs.append(entry)
    SKILL_RUNS_FILE.write_text(json.dumps(runs, indent=2))
