import warnings
warnings.filterwarnings("ignore")

from collections import Counter
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data import DEMO_SERIES
from agents.profiler import profile_series, describe_profile
from agents.tournament import (
    run_tournament, run_final_forecast, get_holdout_forecasts,
    MODEL_COLORS, MODEL_DESCRIPTIONS,
)
from agents.memory import save_lesson, retrieve_similar, cognee_synthesis
from agents.reflection import generate_lesson
from agents.skill_runner import log_skill_run, load_skill_runs
from agents.skill_updater import update_skill
from agents.claude_agent import run_agent

st.set_page_config(page_title="MemoryCast", page_icon="🧠", layout="wide")

st.title("🧠 MemoryCast")
st.caption("A self-improving forecasting agent that learns which strategies work best over time.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Forecast Settings")
    series_name = st.selectbox("Select time series", list(DEMO_SERIES.keys()))
    horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=6, value=3)
    agent_mode = st.toggle("🤖 Claude Agent mode", value=True,
                           help="Claude reads SKILL.md, reasons about memory, and decides which models to run")
    run = st.button("▶ Run Forecast", type="primary", use_container_width=True)
    st.divider()
    st.caption(
        "**Demo story:**\n"
        "1. *Retail Demand* → ETS wins on seasonal data\n"
        "2. *Support Volume* → ML wins on volatile data\n"
        "3. *Cash Inflow* → memory retrieved, mixed past results → tournament decides\n"
        "4. *Skill Evolution tab* → see SKILL.md improve from run history"
    )

df = DEMO_SERIES[series_name]
profile = profile_series(df)

tab_forecast, tab_skill = st.tabs(["📈 Forecast", "🧠 Skill Evolution"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_forecast:

    # ── Series Profile + Memory ───────────────────────────────────────────────
    col_profile, col_memory = st.columns(2)

    with col_profile:
        st.subheader("📊 Series Profile")
        st.caption(describe_profile(profile))

        metric_cols = st.columns(3)
        metric_cols[0].metric("Volatility (CV)", f"{profile['volatility']:.2f}")
        metric_cols[1].metric("Seasonality", f"{profile['seasonality_strength']:.2f}")
        metric_cols[2].metric("Trend R²", f"{profile['trend_strength']:.2f}")

        fig_hist = px.line(df, x="ds", y="y", title="Historical series")
        fig_hist.update_layout(height=220, margin=dict(t=30, b=0, l=0, r=0), showlegend=False)
        fig_hist.update_traces(line_color="#4f8ef7")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_memory:
        st.subheader("🗂️ Memory Retrieved")
        memories = retrieve_similar(profile, top_k=3)
        if memories:
            for mem in memories:
                sp = mem.get("series_profile", {})
                diff = (
                    abs(sp.get("volatility", 0) - profile["volatility"])
                    + abs(sp.get("seasonality_strength", 0) - profile["seasonality_strength"])
                    + abs(sp.get("trend_strength", 0) - profile["trend_strength"])
                )
                similarity = max(0, round((1 - diff / 3) * 100))
                source = mem.get("series_label") or "past series"
                model = mem.get("winning_model", "?")
                wape = mem.get("winner_wape", 0)
                with st.expander(f"📌 {source} — **{model}** won · {similarity}% match", expanded=True):
                    st.markdown(mem.get("lesson_text", ""))
                    st.caption(f"WAPE: {wape:.1%} · {mem.get('saved_at', '')[:10]}")

            winner_counts = Counter(m.get("winning_model") for m in memories)
            top_model, top_count = winner_counts.most_common(1)[0]
            total = len(memories)
            if top_count > 1:
                st.success(f"💡 **{top_count}/{total} similar past runs** favoured **{top_model}**")
            else:
                st.warning(f"💡 **Mixed results** across {total} similar past runs — full tournament will decide")

            # Cognee knowledge-graph synthesis
            profile_desc = (
                f"volatility={profile['volatility']:.2f} "
                f"seasonality={profile['seasonality_strength']:.2f} "
                f"trend={profile['trend_strength']:.2f} "
                f"forecasting model recommendation"
            )
            with st.spinner("🔮 Querying Cognee knowledge graph…"):
                synthesis = cognee_synthesis(profile_desc)
            if synthesis:
                st.info(f"**Cognee synthesis:** {synthesis}")
                st.caption("↑ LLM-synthesised from the Cognee knowledge graph (vectors in Moss)")
        else:
            st.info("No prior memory found — running full model tournament for the first time.")

    st.divider()

    # ── Run ───────────────────────────────────────────────────────────────────
    if run:

        with st.status("🤖 Running forecast agent…", expanded=True) as agent_status:

            if agent_mode:
                # ── Claude Agent mode ─────────────────────────────────────────
                st.write("**Claude Agent reading SKILL.md and memory…**")
                agent_steps = st.empty()
                step_log = []

                def on_step(label, content):
                    step_log.append(f"**{label}**")
                    if content and len(content) < 400:
                        step_log.append(f"↳ {content}")
                    agent_steps.markdown("\n\n".join(step_log[-12:]))

                agent_result = run_agent(
                    series_name, horizon=horizon,
                    df_store=DEMO_SERIES, on_step=on_step,
                )
                results = agent_result["tournament_results"]
                winner = agent_result["winner"]
                lesson_text = agent_result["reasoning"] or ""

                # Build a lesson dict for the UI (save_result already called inside agent)
                lesson = {
                    "lesson_text": lesson_text,
                    "winning_model": winner,
                    "winner_wape": results[winner]["wape"] if results else 0,
                    "improvement_pp": round(
                        (results.get("SeasonalNaive", {}).get("wape", 0) - results[winner]["wape"]) * 100, 1
                    ) if results else 0,
                    "recommendation": f"Use {winner} for {describe_profile(profile)}",
                }
                skill_entry = load_skill_runs()[-1] if load_skill_runs() else {}

            else:
                # ── Manual pipeline mode ──────────────────────────────────────
                st.write("**Step 1 — Series profiled**")
                st.write(f"↳ {describe_profile(profile)}")

                st.write("**Step 2 — Memory retrieved**")
                if memories:
                    for m in memories:
                        st.write(f"↳ [{m.get('series_label','?')}] {m.get('winning_model','?')} won · WAPE {m.get('winner_wape',0):.1%}")
                else:
                    st.write("↳ No prior memory — exploring all strategies cold")

                st.write("**Step 3 — Feature recipes**")
                for model, desc in MODEL_DESCRIPTIONS.items():
                    st.write(f"↳ **{model}**: {desc}")

                st.write("**Step 4 — Running tournament** (3-window backtest cross-validation)…")
                results, winner = run_tournament(df, horizon=horizon)
                for model, res in sorted(results.items(), key=lambda x: x[1]["wape"]):
                    icon = "🏆" if model == winner else "  "
                    st.write(f"↳ {icon} **{model}**: WAPE={res['wape']:.1%}  bias={res['bias']:+.1%}")

                st.write(f"**Step 5 — Winner: {winner}**")
                st.write("**Step 6 — Saving lesson + SkillRunEntry to memory**")
                lesson = generate_lesson(profile, results, winner, series_label=series_name)
                save_lesson(lesson)
                skill_entry = log_skill_run(series_name, profile, results, winner, lesson["lesson_text"])
                st.write(f"↳ {lesson['lesson_text']}")
                st.write(f"↳ SkillRunEntry logged: score={skill_entry['success_score']:.2f}, improvement=+{skill_entry['improvement']*100:.1f}pp vs baseline")

            st.write("**Computing holdout forecasts and future forecast…**")
            holdout_forecasts, actuals = get_holdout_forecasts(df, horizon)
            forecast_df = run_final_forecast(df, horizon, winner)

            agent_status.update(
                label=f"✅ Agent complete — **{winner}** won · {results[winner]['wape']:.1%} WAPE"
                      + (f" · score {skill_entry.get('success_score', 0):.2f}" if skill_entry else ""),
                state="complete",
            )

        st.divider()

        # ── Holdout comparison ────────────────────────────────────────────────
        st.subheader("📈 Model Comparison — Holdout Period")
        st.caption(
            f"Each model trained on first {len(df) - horizon} months, forecasted the last "
            f"{horizon} months (actuals known). Winner is the closest line to green."
        )

        train_df = df.iloc[:-horizon]
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=train_df["ds"], y=train_df["y"],
            name="History", line=dict(color="#4f8ef7", width=2),
        ))
        fig_comp.add_trace(go.Scatter(
            x=actuals["ds"], y=actuals["actual"],
            name="Actuals (holdout)", mode="lines+markers",
            line=dict(color="#22c55e", width=3),
            marker=dict(size=8),
        ))
        for model, fc_df in holdout_forecasts.items():
            is_winner = model == winner
            color = MODEL_COLORS.get(model, "#888")
            fig_comp.add_trace(go.Scatter(
                x=fc_df["ds"], y=fc_df["forecast"],
                name=f"{model} {'🏆' if is_winner else ''} (WAPE {results[model]['wape']:.1%})",
                mode="lines+markers",
                line=dict(color=color, width=3 if is_winner else 1.5, dash="solid" if is_winner else "dot"),
                marker=dict(size=7 if is_winner else 4),
                opacity=1.0 if is_winner else 0.45,
            ))
        fig_comp.update_layout(
            height=420, margin=dict(t=20, b=0, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        table_df = pd.DataFrame([
            {"Model": k, "Type": v["type"], "WAPE": f"{v['wape']:.1%}",
             "Bias": f"{v['bias']:+.1%}", "Score": f"{1-v['wape']:.2f}", "Winner": "🏆" if k == winner else ""}
            for k, v in sorted(results.items(), key=lambda x: x[1]["wape"])
        ])
        st.dataframe(table_df, hide_index=True, use_container_width=True)

        st.divider()

        # ── Future forecast ───────────────────────────────────────────────────
        st.subheader(f"🔮 Future Forecast — Next {horizon} Months ({winner})")
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="History", line=dict(color="#4f8ef7", width=2)))
        fig_fc.add_trace(go.Scatter(
            x=[df["ds"].iloc[-1], forecast_df["ds"].iloc[0]],
            y=[df["y"].iloc[-1], forecast_df["forecast"].iloc[0]],
            line=dict(color="#f59e0b", width=2, dash="dash"), showlegend=False,
        ))
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["ds"], y=forecast_df["forecast"],
            name=f"Forecast ({winner})", mode="lines+markers",
            line=dict(color="#f59e0b", width=3, dash="dash"), marker=dict(size=8),
        ))
        fig_fc.update_layout(height=320, margin=dict(t=20, b=0, l=0, r=0))
        st.plotly_chart(fig_fc, use_container_width=True)

        st.divider()

        # ── Lesson ────────────────────────────────────────────────────────────
        st.subheader("💾 Lesson Saved to Memory")
        l_col, m_col = st.columns([3, 1])
        with l_col:
            st.markdown(f"> {lesson.get('lesson_text', '')[:400]}")
            st.caption(f"**Stored recommendation:** {lesson.get('recommendation', '')}")
        with m_col:
            st.metric("Winner WAPE", f"{lesson.get('winner_wape', results[winner]['wape']):.1%}")
            st.metric("vs Best Loser", f"−{lesson.get('improvement_pp', 0)}pp")
        st.success("✅ Lesson saved — the agent will use this on the next similar series.")

    else:
        st.info("Configure settings in the sidebar and click **▶ Run Forecast** to start.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SKILL EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_skill:
    st.subheader("🧠 Skill Self-Improvement")
    st.caption(
        "The agent's forecasting strategy is encoded in **SKILL.md**. "
        "After accumulating SkillRunEntry records, the skill updater rewrites it with learned rules. "
        "The diff shows exactly what the agent learned."
    )

    runs = load_skill_runs()

    # ── Run history table ─────────────────────────────────────────────────────
    st.markdown(f"#### SkillRunEntry Records ({len(runs)} runs)")
    if runs:
        runs_df = pd.DataFrame([
            {
                "Run ID": r["run_id"],
                "Series": r["series_label"],
                "Strategy Used": r["strategy_used"],
                "Success Score": f"{r['success_score']:.2f}",
                "WAPE": f"{r['winner_wape']:.1%}",
                "vs Baseline": f"+{r['improvement']*100:.1f}pp",
                "Feedback": r["feedback"],
            }
            for r in runs
        ])
        st.dataframe(runs_df, hide_index=True, use_container_width=True)

        avg_score = sum(r["success_score"] for r in runs) / len(runs)
        avg_imp = sum(r["improvement"] for r in runs) / len(runs) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Runs", len(runs))
        m2.metric("Avg Success Score", f"{avg_score:.2f}")
        m3.metric("Avg Improvement vs Baseline", f"+{avg_imp:.1f}pp")
    else:
        st.info("No SkillRunEntry records yet. Run forecasts in the Forecast tab first.")

    st.divider()

    # ── Skill diff ────────────────────────────────────────────────────────────
    st.markdown("#### SKILL.md — Before vs After")

    if runs:
        if st.button("🔄 Update Skill from Run History", type="primary"):
            v1, v2, diff = update_skill()

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown("**v1 — Baseline (no experience)**")
                st.code(v1, language="markdown")
            with col_v2:
                st.markdown("**v2 — Learned (from run history)**")
                st.code(v2, language="markdown")

            st.markdown("**Unified Diff — what the agent learned:**")
            st.code(diff, language="diff")

            st.success(
                f"✅ Skill updated from {len(runs)} SkillRunEntry records. "
                "The agent now has concrete rules backed by evidence."
            )
        else:
            from pathlib import Path
            skill_text = Path("skills/memorycast/SKILL.md").read_text()
            st.markdown("**Current SKILL.md (v1 baseline):**")
            st.code(skill_text, language="markdown")
            st.info("Click **Update Skill** to generate v2 from run history and show the diff.")
    else:
        from pathlib import Path
        skill_text = Path("skills/memorycast/SKILL.md").read_text()
        st.markdown("**Current SKILL.md (v1 baseline — no runs yet):**")
        st.code(skill_text, language="markdown")
