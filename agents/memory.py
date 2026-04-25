"""
Memory store — JSON primary, Cognee knowledge graph secondary.

All Cognee async calls run on a single persistent event loop (cognee-worker thread)
to avoid "Lock bound to different event loop" errors from re-using module-level
asyncio primitives across multiple asyncio.run() calls.
"""

import json
import os
import asyncio
import threading
from datetime import datetime
from pathlib import Path

MEMORY_FILE = Path(__file__).parent.parent / "memory" / "lessons.json"

# ── Single persistent Cognee event loop ──────────────────────────────────────

_cognee_loop: asyncio.AbstractEventLoop | None = None
_cognee_worker: threading.Thread | None = None
_cognee_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _cognee_loop, _cognee_worker
    with _cognee_lock:
        if _cognee_loop is None or _cognee_loop.is_closed():
            _cognee_loop = asyncio.new_event_loop()
            _cognee_worker = threading.Thread(
                target=_cognee_loop.run_forever,
                daemon=True,
                name="cognee-worker",
            )
            _cognee_worker.start()
    return _cognee_loop


def _fire(coro) -> "asyncio.Future":
    """Schedule coroutine on the Cognee worker loop (fire and forget)."""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop())


def _run_sync(coro, timeout: float = 30.0):
    """Schedule coroutine on the Cognee worker loop and wait for result."""
    return _fire(coro).result(timeout=timeout)


# ── Cognee helpers ────────────────────────────────────────────────────────────

def _cognee_setup():
    import cognee
    cognee.config.set_llm_api_key(os.environ["OPENAI_API_KEY"])
    cognee.config.set_llm_model("gpt-4o-mini")
    cognee.config.set_embedding_provider("openai")
    cognee.config.set_embedding_model("text-embedding-3-small")
    cognee.config.set_embedding_dimensions(1536)


def _format_lesson_text(lesson: dict) -> str:
    sp = lesson.get("series_profile", {})
    return (
        f"MemoryCast forecasting lesson. {lesson.get('lesson_text', '')} "
        f"Series: {lesson.get('series_label', 'unknown')}. "
        f"Profile: volatility={sp.get('volatility', 0):.2f}, "
        f"seasonality={sp.get('seasonality_strength', 0):.2f}, "
        f"trend={sp.get('trend_strength', 0):.2f}. "
        f"Winner: {lesson.get('winning_model', '?')} "
        f"WAPE={lesson.get('winner_wape', 0):.1%}. "
        f"Recommendation: {lesson.get('recommendation', '')}."
    )


async def _remember(text: str):
    import cognee
    await cognee.remember(text)


async def _recall_synthesis(query: str) -> str | None:
    import cognee
    results = await cognee.recall(query)
    if results:
        sr = results[0].get("search_result", [])
        return sr[0] if sr else None
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def save_lesson(lesson: dict) -> None:
    """Persist a forecasting lesson to JSON and queue a Cognee remember."""
    MEMORY_FILE.parent.mkdir(exist_ok=True)
    lessons = _load()
    lesson["saved_at"] = datetime.utcnow().isoformat()
    lessons.append(lesson)
    MEMORY_FILE.write_text(json.dumps(lessons, indent=2))

    _cognee_setup()
    _fire(_remember(_format_lesson_text(lesson)))


def retrieve_similar(profile: dict, top_k: int = 3) -> list[dict]:
    """Return top-k lessons most similar to the current series profile."""
    lessons = _load()
    if not lessons:
        return []
    scored = [(_similarity(profile, l.get("series_profile", {})), l) for l in lessons]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [l for _, l in scored[:top_k]]


def cognee_synthesis(profile_desc: str, timeout: float = 25.0) -> str | None:
    """
    Ask Cognee to synthesize relevant memory for this series profile.
    Returns the synthesis string, or None if unavailable / timed out.
    """
    try:
        _cognee_setup()
        return _run_sync(_recall_synthesis(profile_desc), timeout=timeout)
    except Exception:
        return None


def seed_cognee(texts: list[str], timeout_per_item: float = 60.0):
    """
    Block until every text in `texts` is remembered by Cognee.
    Used by seed_memory.py to pre-populate the knowledge graph.
    """
    _cognee_setup()
    futures = [_fire(_remember(t)) for t in texts]
    for i, f in enumerate(futures, 1):
        f.result(timeout=timeout_per_item)
        print(f"  Cognee remembered item {i}/{len(futures)}")


# ── Private helpers ───────────────────────────────────────────────────────────

def _load() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    return json.loads(MEMORY_FILE.read_text())


def _similarity(a: dict, b: dict) -> float:
    keys = ["volatility", "seasonality_strength", "trend_strength"]
    diffs = [abs(a.get(k, 0) - b.get(k, 0)) for k in keys if k in a and k in b]
    return 1 - (sum(diffs) / len(diffs)) if diffs else 0.0
