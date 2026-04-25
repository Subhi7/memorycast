"""
Reflection agent — generates a reusable forecasting lesson using Claude.
"""
import os
from dotenv import load_dotenv
load_dotenv()

import anthropic

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def generate_lesson(profile: dict, results: dict, winner: str, series_label: str = "") -> dict:
    winner_wape = results[winner]["wape"]
    losers = {k: v["wape"] for k, v in results.items() if k != winner}
    worst_loser = max(losers, key=losers.get)
    improvement_pp = round((losers[worst_loser] - winner_wape) * 100, 1)

    lesson_text = _generate_with_claude(profile, results, winner, winner_wape, losers, series_label)

    return {
        "series_label": series_label,
        "series_profile": {
            "volatility": profile["volatility"],
            "seasonality_strength": profile["seasonality_strength"],
            "trend_strength": profile["trend_strength"],
            "history_length": profile["history_length"],
        },
        "winning_model": winner,
        "winner_wape": winner_wape,
        "losing_models": losers,
        "improvement_pp": improvement_pp,
        "lesson_text": lesson_text,
        "recommendation": f"Use {winner} for series with volatility≈{profile['volatility']:.2f}, seasonality≈{profile['seasonality_strength']:.2f}.",
    }


def _generate_with_claude(profile, results, winner, winner_wape, losers, series_label):
    vol = "high" if profile["volatility"] > 0.15 else "low"
    seas = "strong" if profile["seasonality_strength"] > 0.4 else "weak"
    worst_loser = max(losers, key=losers.get)
    improvement_pp = round((losers[worst_loser] - winner_wape) * 100, 1)

    ranked = ", ".join(f"{k}={v['wape']:.1%}" for k, v in sorted(results.items(), key=lambda x: x[1]["wape"]))
    extra = {}
    if hasattr(profile, "get"):
        extra = {k: profile.get(k) for k in ("acf_lag1", "seasonal_amplitude", "regime_shift", "trend_direction", "outlier_rate") if profile.get(k) is not None}

    prompt = f"""You are a forecasting expert writing a reusable memory entry for an AI agent.

Tournament run on '{series_label or 'unknown'}':
- Profile: {vol} volatility (CV={profile['volatility']:.2f}), {seas} seasonality ({profile['seasonality_strength']:.2f}), trend R²={profile['trend_strength']:.2f}{(', ' + ', '.join(f'{k}={v}' for k,v in extra.items())) if extra else ''}
- Results (WAPE): {ranked}
- Winner: **{winner}** at {winner_wape:.1%} WAPE — {improvement_pp}pp better than {worst_loser}

Write a memory entry in this exact format (3 lines, no extra text):
**Why {winner} won:** <one sentence on which profile signals drove this result>
**When to reuse:** <one sentence on the profile conditions where {winner} should be the first choice>
**Watch out for:** <one sentence on when this rule might fail>"""

    try:
        response = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception:
        return _fallback_text(profile, winner, winner_wape, worst_loser, improvement_pp)


def _fallback_text(profile, winner, winner_wape, worst_loser, improvement_pp):
    vol = "high" if profile["volatility"] > 0.15 else "low"
    seas = "strong" if profile["seasonality_strength"] > 0.4 else "weak"
    if winner in ("AutoARIMA", "AutoETS"):
        reason = f"Statistical model leveraged the {seas} seasonality pattern."
    elif winner == "GradientBoosting":
        reason = f"Lag-based ML handled {vol}-volatility shifts that statistical models missed."
    else:
        reason = f"{winner} was most stable across backtest windows."
    return (
        f"For {vol}-volatility, {seas}-seasonality series, **{winner}** achieved "
        f"{winner_wape:.1%} WAPE — {improvement_pp}pp better than {worst_loser}. {reason}"
    )
