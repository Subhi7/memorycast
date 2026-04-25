# MemoryCast -- Submission

## Team
Subhiksha Mukuntharaj

## Skill folder
`skills/memorycast/SKILL.md`

## What it does
MemoryCast is a self-improving time-series forecasting agent. It profiles each series, queries Cognee memory for similar past runs, runs a model tournament (SeasonalNaive / AutoARIMA / AutoETS / GradientBoosting), and writes learned rules back into SKILL.md -- autonomously, with no human edits.

## Before score (v1 -- 0 runs)
- Avg success score: **0.834**
- Avg improvement over baseline: **2.8pp**
- Rules: 5 manual heuristics, no evidence

## After score (v2 -- 20 runs)
- Avg success score: **0.934** (+10pp)
- Best run: SaaS ARR -- AutoARIMA, 1.3% WAPE
- Rules: 3 learned rules with evidence citations from Cognee

## Skill diff (v1 -> v2)
Key additions from 20 SkillRunEntry records:
- High-vol + strong-seasonality -> AutoETS (5/5 runs, avg 2.9% WAPE)
- High-vol + weak-seasonality -> AutoARIMA (6/10 runs, avg 12.1% WAPE)
- Regime shift -> GradientBoosting (+8.3pp best run)

## Feedback records
`memory/skill_runs.json` -- 20 SkillRunEntry records stored in Cognee / Moss vectors

Sample:
```
run_id=8a40e048 | Retail Demand (Seasonal) | AutoETS | score=0.971
feedback: AutoETS won on 'Retail Demand (Seasonal)' (high-volatility, strong-seasonality).
Profile: CV=0.19, seasonality=0.57. Tournament: AutoETS=2.9%, AutoARIMA=4.1%, SeasonalNaive=4.5%

run_id=a2eb3bf6 | Cash Inflow (Volatile) | AutoARIMA | score=0.906
feedback: AutoARIMA won on 'Cash Inflow (Volatile)' (high-volatility, weak-seasonality).
Profile: CV=0.18, seasonality=0.08. Tournament: AutoARIMA=9.4%, AutoETS=11.2%, SeasonalNaive=17.4%
```

## Daytona run
```bash
uv sync
uv run python seed_memory.py
uv run python agents/skill_updater.py
uv run streamlit run app.py
```

## Self-improvement loop
1. Profile series (volatility, seasonality, trend, ACF, regime shift)
2. Query Cognee (Moss cloud vectors) for top-3 similar past runs
3. Claude agent (7 tools, up to 10 turns) decides which models to run
4. Cross-validation tournament picks winner by lowest WAPE
5. SkillRunEntry saved to Cognee with success_score + feedback
6. skill_updater reads all entries and rewrites SKILL.md with learned rules + evidence
