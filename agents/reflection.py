"""
Reflection agent — generates a reusable forecasting lesson from a tournament result.
Rule-based now; Claude API stub is ready to drop in.
"""


def generate_lesson(profile: dict, results: dict, winner: str) -> dict:
    winner_wape = results[winner]["wape"]
    losers = {k: v["wape"] for k, v in results.items() if k != winner}
    worst_loser = max(losers, key=losers.get)
    improvement_pp = round((losers[worst_loser] - winner_wape) * 100, 1)

    lesson_text = _rule_based_text(profile, winner, winner_wape, worst_loser, improvement_pp)

    return {
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


def _rule_based_text(profile, winner, winner_wape, worst_loser, improvement_pp):
    vol = "high" if profile["volatility"] > 0.15 else "low"
    seas = "strong" if profile["seasonality_strength"] > 0.4 else "weak"

    if winner == "LightGBM":
        reason = (
            f"Lag-based ML captured recent volatility shifts that {worst_loser} missed."
        )
    elif winner in ("AutoARIMA", "AutoETS"):
        reason = (
            f"Statistical model exploited the {seas} seasonality pattern effectively."
        )
    else:
        reason = f"{winner} was the most stable model across backtest windows."

    return (
        f"For {vol}-volatility, {seas}-seasonality series, **{winner}** achieved "
        f"{winner_wape:.1%} WAPE — {improvement_pp}pp better than {worst_loser}. "
        f"{reason}"
    )

    # --- Claude stub ---
    # import anthropic
    # client = anthropic.Anthropic()
    # prompt = f"""
    # You are a forecasting expert. A tournament was just run on a time series.
    # Series profile: {profile}
    # Results: {results}
    # Winner: {winner}
    # Write a 2-sentence reusable lesson explaining WHY {winner} won and when to use it again.
    # """
    # response = client.messages.create(
    #     model="claude-sonnet-4-6",
    #     max_tokens=200,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return response.content[0].text
