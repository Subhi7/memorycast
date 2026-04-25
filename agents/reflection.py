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

    prompt = f"""You are a forecasting expert writing a reusable lesson for an AI agent's memory.

A model tournament was run on a time series with these results:
- Series: {series_label or 'unknown'}
- Profile: {vol} volatility (CV={profile['volatility']:.2f}), {seas} seasonality ({profile['seasonality_strength']:.2f}), trend R²={profile['trend_strength']:.2f}
- Tournament results (WAPE): {', '.join(f"{k}={v['wape']:.1%}" for k, v in sorted(results.items(), key=lambda x: x[1]['wape']))}
- Winner: {winner} with {winner_wape:.1%} WAPE ({improvement_pp}pp better than {worst_loser})

Write ONE concise sentence (max 25 words) explaining WHY {winner} won and when to use it again.
Focus on the relationship between the series profile and model performance. Use markdown bold for the model name."""

    try:
        response = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
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
