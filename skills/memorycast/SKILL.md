# Forecasting Strategy Skill — v2
> Auto-updated from 20 SkillRunEntry records stored in Cognee memory.

## Core Strategy
1. Profile the series (volatility, seasonality, trend strength)
2. Retrieve similar SkillRunEntry records from Cognee memory
3. Apply learned rules below — skip to full tournament if signals are mixed
4. Run cross-validation tournament (3 windows) to confirm
5. Record new SkillRunEntry with success_score and feedback
6. Re-run skill_updater to incorporate new evidence into next version

## Learned Rules (from run history)
- **High volatility + weak seasonality** (volatility > 0.15 AND seasonality < 0.3)
  → Preferred model: **AutoARIMA** (avg success score: 0.90 across 10 run(s))
  → Evidence:
    - Cash Inflow (Volatile) — AutoARIMA won, WAPE 9.4%, score 0.91
    - Cash Inflow (Volatile) — AutoARIMA won, WAPE 9.4%, score 0.91
    - Cash Inflow (Volatile) — AutoARIMA won, WAPE 9.4%, score 0.91

- **Low volatility + strong seasonality** (volatility ≤ 0.15 AND seasonality ≥ 0.4)
  → Preferred model: **AutoARIMA** (avg success score: 0.99 across 3 run(s))
  → Evidence:
    - SaaS Annual Recurring Revenue — AutoARIMA won, WAPE 1.3%, score 0.99
    - E-commerce Orders — AutoETS won, WAPE 1.6%, score 0.98
    - Seasonal Ad Spend — GradientBoosting won, WAPE 3.0%, score 0.97

- **High volatility + strong seasonality** (volatility > 0.15 AND seasonality ≥ 0.4)
  → Preferred model: **AutoETS** (avg success score: 0.97 across 7 run(s))
  → Evidence:
    - Retail Demand (Seasonal) — AutoETS won, WAPE 2.9%, score 0.97
    - Retail Demand (Seasonal) — AutoETS won, WAPE 2.9%, score 0.97
    - Retail Demand (Seasonal) — AutoETS won, WAPE 2.9%, score 0.97

## Fallback Rule
When fewer than 2 similar past runs exist, run the full tournament:
SeasonalNaive → AutoARIMA → AutoETS → GradientBoosting
Select by lowest cross-validation WAPE.

## Feature Engineering (GradientBoosting)
- Lag features: 1, 2, 3, 6, 12 months back
- Date features: month of year
- Evidence suggests lag-12 is critical for annual patterns

## Run History Summary
Total runs: 20 | Avg improvement over baseline: 3.1pp | Best run: SaaS Annual Recurring Revenue (AutoARIMA, 1.3% WAPE)

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
