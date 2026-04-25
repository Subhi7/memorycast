# Forecasting Strategy Skill — v1
> Baseline skill. No prior experience. All rules are manually defined.

## Core Strategy
1. Profile the series (volatility, seasonality, trend strength)
2. If no memory: run full model tournament
3. Prefer statistical models for strong-seasonality series
4. Prefer ML lag models for high-volatility series
5. Select winner by lowest WAPE across 3-window cross-validation
6. Save lesson to Cognee memory for future reference

## Models Available

| Model | Type | Best for |
|-------|------|----------|
| SeasonalNaive | Statistical | Stable seasonal baseline |
| AutoARIMA | Statistical | Seasonal + trend patterns |
| AutoETS | Statistical | Error/Trend/Seasonality smoothing |
| GradientBoosting | ML | Volatile, nonlinear, regime-shift series |

## Feature Engineering (GradientBoosting)
- Lag features: 1, 2, 3, 6, 12 months back
- Date features: month of year
- No exogenous variables in v1

## Success Metric
- Primary: WAPE (Weighted Absolute Percentage Error)
- Secondary: Bias (signed over/under-forecast direction)
- `success_score = 1 - winner_wape`
- `improvement = baseline_wape - winner_wape` (vs SeasonalNaive)

## Self-Improvement Loop
1. Each run produces a SkillRunEntry stored in Cognee
2. After N runs, skill_updater reads all SkillRunEntry records
3. Patterns are extracted — which model wins for which profile type
4. Learned rules are written into SKILL.md v2
5. Next agent run uses the improved skill

## Limitations (v1)
- No learned rules yet — relies on general heuristics only
- Does not adapt to domain-specific patterns
- No confidence weighting across run history
