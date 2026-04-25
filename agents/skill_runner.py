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
        "feedback": f"{winner} beat SeasonalNaive baseline by {improvement * 100:.1f}pp WAPE",
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
