"""
Claude-powered forecasting agent.

Claude reads SKILL.md, retrieves memory, decides which models to run
(can skip low-confidence models based on strong priors), interprets
results, and saves the lesson — all via tool calls.
"""

import json
import os
from pathlib import Path

import anthropic

from agents.profiler import profile_series, describe_profile
from agents.tournament import run_tournament, MODEL_DESCRIPTIONS
from agents.memory import retrieve_similar, save_lesson, cognee_synthesis
from agents.reflection import generate_lesson
from agents.skill_runner import log_skill_run

SKILL_FILE = Path(__file__).parent.parent / "skills" / "memorycast" / "SKILL.md"
MAX_TURNS = 10

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_skill",
        "description": "Read the current SKILL.md forecasting strategy. Always call this first to understand your strategy rules and any learned rules from past runs.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "profile_series",
        "description": (
            "Compute a rich profile for the series. Returns: volatility (CV), seasonality_strength (ACF lag-12), "
            "trend_strength (R²), trend_direction (up/down/flat), trend_slope_pct (% per month), "
            "peak_month (1-12), seasonal_amplitude, acf_lag1 (AR signal strength), recent_growth, "
            "yoy_change, outlier_rate, skewness, is_stationary, regime_shift, mean, std."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_name": {"type": "string", "description": "Name of the series to profile"},
            },
            "required": ["series_name"],
        },
    },
    {
        "name": "retrieve_memory",
        "description": "Retrieve the top-3 most similar past forecasting runs from memory. Use the series profile to find relevant precedents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "volatility": {"type": "number"},
                "seasonality_strength": {"type": "number"},
                "trend_strength": {"type": "number"},
            },
            "required": ["volatility", "seasonality_strength", "trend_strength"],
        },
    },
    {
        "name": "cognee_recall",
        "description": "Query the Cognee knowledge graph for a synthesised recommendation. Use when memory cards are mixed or inconclusive.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query about the series profile and which model to use"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_full_tournament",
        "description": "Run all 4 models (SeasonalNaive, AutoARIMA, AutoETS, GradientBoosting) with 3-window cross-validation. Use when memory is mixed or insufficient.",
        "input_schema": {
            "type": "object",
            "properties": {
                "series_name": {"type": "string"},
                "horizon": {"type": "integer", "description": "Forecast horizon in months (1-6)", "default": 3},
            },
            "required": ["series_name"],
        },
    },
    {
        "name": "run_focused_tournament",
        "description": "Run only specified models. Use when SKILL.md learned rules or strong memory consensus (3/3) suggests skipping low-confidence models.",
        "input_schema": {
            "type": "object",
            "properties": {
                "series_name": {"type": "string"},
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Subset of models to run. Must include SeasonalNaive as baseline.",
                },
                "horizon": {"type": "integer", "default": 3},
                "reason": {"type": "string", "description": "Why these models were selected"},
            },
            "required": ["series_name", "models"],
        },
    },
    {
        "name": "save_result",
        "description": "Save the forecasting lesson and SkillRunEntry to Cognee memory. Always call this after you have tournament results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "series_name": {"type": "string"},
                "winner": {"type": "string"},
                "agent_reasoning": {
                    "type": "string",
                    "description": (
                        "Write exactly 3 lines in this format:\n"
                        "**Why <winner> won:** <one sentence on which profile signals drove this>\n"
                        "**When to reuse:** <one sentence on profile conditions for this model>\n"
                        "**Watch out for:** <one sentence on when this rule might fail>"
                    ),
                },
            },
            "required": ["series_name", "winner", "agent_reasoning"],
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────

def _tool_get_skill() -> str:
    return SKILL_FILE.read_text()


def _tool_profile_series(series_name: str, df_store: dict) -> dict:
    df = df_store[series_name]
    profile = profile_series(df)
    return {
        "volatility": round(profile["volatility"], 3),
        "seasonality_strength": round(profile["seasonality_strength"], 3),
        "trend_strength": round(profile["trend_strength"], 3),
        "history_length": profile["history_length"],
        "description": describe_profile(profile),
    }


def _tool_retrieve_memory(vol: float, seas: float, trend: float) -> list[dict]:
    profile = {"volatility": vol, "seasonality_strength": seas, "trend_strength": trend}
    memories = retrieve_similar(profile, top_k=3)
    out = []
    for m in memories:
        sp = m.get("series_profile", {})
        diff = (
            abs(sp.get("volatility", 0) - vol)
            + abs(sp.get("seasonality_strength", 0) - seas)
            + abs(sp.get("trend_strength", 0) - trend)
        )
        similarity = max(0, round((1 - diff / 3) * 100))
        out.append({
            "series": m.get("series_label", "?"),
            "winner": m.get("winning_model", "?"),
            "wape": round(m.get("winner_wape", 0), 4),
            "similarity_pct": similarity,
            "lesson": m.get("lesson_text", ""),
        })
    return out


def _tool_cognee_recall(query: str) -> str:
    result = cognee_synthesis(query)
    return result or "No synthesis available from Cognee knowledge graph."


def _tool_run_full_tournament(series_name: str, horizon: int, df_store: dict) -> dict:
    df = df_store[series_name]
    results, winner = run_tournament(df, horizon=horizon)
    ranked = sorted(results.items(), key=lambda x: x[1]["wape"])
    return {
        "winner": winner,
        "ranked": [{"model": m, "wape": round(v["wape"], 4), "bias": round(v["bias"], 4)} for m, v in ranked],
        "winner_wape": results[winner]["wape"],
        "baseline_wape": results.get("SeasonalNaive", {}).get("wape", 1.0),
        "all_results": results,
    }


def _tool_run_focused_tournament(
    series_name: str, models: list[str], horizon: int, reason: str, df_store: dict
) -> dict:
    df = df_store[series_name]
    # Always include SeasonalNaive as baseline
    if "SeasonalNaive" not in models:
        models = ["SeasonalNaive"] + models
    results, winner = run_tournament(df, horizon=horizon, models=models)
    ranked = sorted(results.items(), key=lambda x: x[1]["wape"])
    return {
        "winner": winner,
        "models_run": models,
        "reason": reason,
        "ranked": [{"model": m, "wape": round(v["wape"], 4), "bias": round(v["bias"], 4)} for m, v in ranked],
        "winner_wape": results[winner]["wape"],
        "baseline_wape": results.get("SeasonalNaive", {}).get("wape", 1.0),
        "all_results": results,
        "skipped_full_tournament": len(models) < 4,
    }


def _tool_save_result(
    series_name: str,
    winner: str,
    agent_reasoning: str,
    tournament_results: dict,
    profile: dict,
    df_store: dict,
) -> dict:
    df = df_store[series_name]
    p = profile_series(df)
    lesson = generate_lesson(p, tournament_results, winner, series_label=series_name)
    lesson["lesson_text"] = agent_reasoning  # Use Claude's own reasoning as the lesson
    save_lesson(lesson)
    entry = log_skill_run(series_name, p, tournament_results, winner, agent_reasoning)
    return {
        "saved": True,
        "run_id": entry["run_id"],
        "success_score": entry["success_score"],
        "improvement_pp": round(entry["improvement"] * 100, 1),
        "feedback": entry["feedback"],
    }


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(series_name: str, horizon: int = 3, df_store: dict = None, on_step=None) -> dict:
    """
    Run the Claude forecasting agent on a named series.

    Args:
        series_name: key in df_store
        horizon: forecast horizon in months
        df_store: dict of {name: DataFrame}
        on_step: optional callback(step_label, content) for streaming to UI

    Returns:
        dict with winner, wape, reasoning, skill_run_entry, steps
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    system = f"""You are a self-improving forecasting agent. Your job is to forecast '{series_name}'.

Follow this process:
1. Call get_skill to read your strategy and any learned rules from past runs.
2. Call profile_series — use ALL fields to reason about the series:
   - high acf_lag1 (>0.7) → strong AR structure → AutoARIMA likely good
   - regime_shift > 1.5 → structural break → GradientBoosting handles non-stationarity better
   - seasonal_amplitude > 0.3 → strong seasonal swing → AutoETS or SeasonalNaive competitive
   - outlier_rate > 0.1 → noisy data → ML lag features more robust
   - is_stationary=False → differencing needed → ARIMA family preferred
   - trend_direction=up/down + trend_strength > 0.6 → AutoARIMA or AutoETS with trend component
3. Call retrieve_memory to find similar past runs.
4. Decide which models to run:
   - Strong memory consensus (3/3) + profile confirms → run_focused_tournament (2-3 models)
   - Mixed memory or new profile type → run_full_tournament
5. Call save_result — write a lesson that explains the profile signals, why the winner won,
   and what future agents should look for on similar series.

Be specific about which profile signals drove your model choice."""

    messages = [{"role": "user", "content": f"Forecast '{series_name}' with a {horizon}-month horizon. Series name for all tool calls: '{series_name}'"}]

    steps = []
    last_tournament_results = None
    last_profile = None

    def _emit(label, content):
        steps.append({"label": label, "content": content})
        if on_step:
            on_step(label, content)

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        # Collect text from this response
        for block in response.content:
            if hasattr(block, "text") and block.text.strip():
                _emit(f"Agent thinking (turn {turn + 1})", block.text.strip())

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            break

        # Process all tool calls in this turn
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            name = block.name
            inp = block.input
            _emit(f"Tool call: {name}", json.dumps(inp, indent=2))

            try:
                if name == "get_skill":
                    result = _tool_get_skill()

                elif name == "profile_series":
                    result = _tool_profile_series(inp["series_name"], df_store)
                    last_profile = result

                elif name == "retrieve_memory":
                    result = _tool_retrieve_memory(
                        inp["volatility"], inp["seasonality_strength"], inp["trend_strength"]
                    )

                elif name == "cognee_recall":
                    result = _tool_cognee_recall(inp["query"])

                elif name == "run_full_tournament":
                    result = _tool_run_full_tournament(
                        inp["series_name"], inp.get("horizon", horizon), df_store
                    )
                    last_tournament_results = result["all_results"]
                    _emit(
                        "Tournament results",
                        " | ".join(f"{r['model']} {r['wape']:.1%}" for r in result["ranked"]),
                    )

                elif name == "run_focused_tournament":
                    result = _tool_run_focused_tournament(
                        inp["series_name"],
                        inp["models"],
                        inp.get("horizon", horizon),
                        inp.get("reason", ""),
                        df_store,
                    )
                    last_tournament_results = result["all_results"]
                    skipped = result.get("skipped_full_tournament", False)
                    _emit(
                        "Focused tournament" + (" (memory-guided, skipped low-confidence models)" if skipped else ""),
                        " | ".join(f"{r['model']} {r['wape']:.1%}" for r in result["ranked"]),
                    )

                elif name == "save_result":
                    if last_tournament_results is None:
                        result = {"error": "No tournament results yet — run a tournament first."}
                    else:
                        result = _tool_save_result(
                            inp["series_name"],
                            inp["winner"],
                            inp["agent_reasoning"],
                            last_tournament_results,
                            last_profile or {},
                            df_store,
                        )
                        _emit(
                            "Saved to memory",
                            f"run_id={result['run_id']} score={result['success_score']:.2f} improvement=+{result['improvement_pp']}pp",
                        )

                else:
                    result = {"error": f"Unknown tool: {name}"}

            except Exception as e:
                result = {"error": str(e)}

            result_str = json.dumps(result) if not isinstance(result, str) else result
            _emit(f"Tool result: {name}", result_str[:500])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
            })

        # Append assistant turn + tool results to conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    # Extract final winner from last tournament
    winner = None
    wape = None
    if last_tournament_results:
        winner = min(last_tournament_results, key=lambda m: last_tournament_results[m]["wape"])
        wape = last_tournament_results[winner]["wape"]

    # Get agent's final text (last assistant text block)
    reasoning = ""
    for block in response.content:
        if hasattr(block, "text"):
            reasoning = block.text.strip()

    return {
        "winner": winner,
        "wape": wape,
        "reasoning": reasoning,
        "tournament_results": last_tournament_results,
        "steps": steps,
        "turns": turn + 1,
    }
