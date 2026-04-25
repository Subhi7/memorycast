"""
Memory store — JSON-backed now, Cognee-ready later.

To swap in Cognee, replace the body of save_lesson() and retrieve_similar()
with cognee.add() / cognee.cognify() / cognee.search() calls.
"""

import json
from datetime import datetime
from pathlib import Path

MEMORY_FILE = Path(__file__).parent.parent / "memory" / "lessons.json"


def save_lesson(lesson: dict) -> None:
    """Persist a forecasting lesson."""
    MEMORY_FILE.parent.mkdir(exist_ok=True)
    lessons = _load()
    lesson["saved_at"] = datetime.utcnow().isoformat()
    lessons.append(lesson)
    MEMORY_FILE.write_text(json.dumps(lessons, indent=2))

    # --- Cognee stub ---
    # import cognee, asyncio
    # asyncio.run(_cognee_save(lesson))

# async def _cognee_save(lesson: dict):
#     await cognee.add(json.dumps(lesson))
#     await cognee.cognify()


def retrieve_similar(profile: dict, top_k: int = 3) -> list[dict]:
    """Return top-k lessons most similar to the current series profile."""
    lessons = _load()
    if not lessons:
        return []

    scored = [(_similarity(profile, l.get("series_profile", {})), l) for l in lessons]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [l for _, l in scored[:top_k]]

    # --- Cognee stub ---
    # import cognee, asyncio
    # from cognee.api.v1.search import SearchType
    # query = f"volatility:{profile['volatility']} seasonality:{profile['seasonality_strength']}"
    # results = asyncio.run(cognee.search(SearchType.INSIGHTS, query))
    # return results[:top_k]


def _load() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    return json.loads(MEMORY_FILE.read_text())


def _similarity(a: dict, b: dict) -> float:
    keys = ["volatility", "seasonality_strength", "trend_strength"]
    diffs = [abs(a.get(k, 0) - b.get(k, 0)) for k in keys if k in a and k in b]
    return 1 - (sum(diffs) / len(diffs)) if diffs else 0.0
