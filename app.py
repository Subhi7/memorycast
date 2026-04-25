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
    MODEL_COLORS, MODEL_DESCRIPTIONS, ML_LAGS, ML_DATE_FEATURES,
)
from agents.memory import save_lesson, retrieve_similar
from agents.reflection import generate_lesson

st.set_page_config(page_title="MemoryCast", page_icon="🧠", layout="wide")

st.title("🧠 MemoryCast")
st.caption("A self-improving forecasting agent that learns which strategies work best over time.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Forecast Settings")
    series_name = st.selectbox("Select time series", list(DEMO_SERIES.keys()))
    horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=6, value=3)
    run = st.button("▶ Run Forecast", type="primary", use_container_width=True)
    st.divider()
    st.caption(
        "**Demo story:**\n"
        "1. *Retail Demand* → ETS wins on seasonal data\n"
        "2. *Support Volume* → ML wins on volatile data\n"
        "3. *Cash Inflow* → memory retrieved, mixed past results → tournament decides"
    )

df = DEMO_SERIES[series_name]
profile = profile_series(df)

# ── Top row: Series Profile + Memory ─────────────────────────────────────────
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
        for i, mem in enumerate(memories):
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
            st.success(f"💡 **{top_count}/{total} similar past runs** favoured **{top_model}** — agent will prioritise this")
        else:
            st.warning(f"💡 **Mixed results** across {total} similar past runs — running full tournament to decide")
    else:
        st.info("No prior memory found — running full model tournament for the first time.")

st.divider()

# ── Run ───────────────────────────────────────────────────────────────────────
if run:

    # ── Step 1: Agent reasoning log ──────────────────────────────────────────
    with st.status("🤖 Running forecast agent…", expanded=True) as agent_status:

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

        st.write(f"**Step 5 — Winner: {winner}** (lowest cross-validation WAPE)")

        st.write("**Step 6 — Computing holdout forecasts for visual comparison…**")
        holdout_forecasts, actuals = get_holdout_forecasts(df, horizon)

        st.write("**Step 7 — Generating future forecast with winning model…**")
        forecast_df = run_final_forecast(df, horizon, winner)

        st.write("**Step 8 — Saving lesson to memory**")
        lesson = generate_lesson(profile, results, winner, series_label=series_name)
        save_lesson(lesson)
        st.write(f"↳ {lesson['lesson_text']}")

        agent_status.update(
            label=f"✅ Agent complete — **{winner}** won with {results[winner]['wape']:.1%} WAPE",
            state="complete",
        )

    st.divider()

    # ── Step 2: Holdout comparison — all models vs actuals ───────────────────
    st.subheader("📈 Model Comparison — Holdout Period")
    st.caption(
        f"Each model was trained on the first {len(df) - horizon} months and forecasted the last "
        f"{horizon} months (which we know). This is how we picked the winner."
    )

    train_df = df.iloc[:-horizon]
    fig_comp = go.Figure()

    # Training history
    fig_comp.add_trace(go.Scatter(
        x=train_df["ds"], y=train_df["y"],
        name="History", line=dict(color="#4f8ef7", width=2),
    ))

    # Actuals holdout
    fig_comp.add_trace(go.Scatter(
        x=actuals["ds"], y=actuals["actual"],
        name="Actuals (holdout)", mode="lines+markers",
        line=dict(color="#22c55e", width=3),
        marker=dict(size=8, symbol="circle"),
    ))

    # Each model's forecast — winner highlighted, others muted
    for model, fc_df in holdout_forecasts.items():
        is_winner = model == winner
        color = MODEL_COLORS.get(model, "#888")
        fig_comp.add_trace(go.Scatter(
            x=fc_df["ds"], y=fc_df["forecast"],
            name=f"{model} {'🏆' if is_winner else ''} (WAPE {results[model]['wape']:.1%})",
            mode="lines+markers",
            line=dict(
                color=color,
                width=3 if is_winner else 1.5,
                dash="solid" if is_winner else "dot",
            ),
            marker=dict(size=7 if is_winner else 4),
            opacity=1.0 if is_winner else 0.45,
        ))

    fig_comp.update_layout(
        height=420,
        margin=dict(t=20, b=0, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="", yaxis_title="Value",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # WAPE summary table
    table_df = pd.DataFrame([
        {
            "Model": k,
            "Type": v["type"],
            "WAPE": f"{v['wape']:.1%}",
            "Bias": f"{v['bias']:+.1%}",
            "Winner": "🏆" if k == winner else "",
        }
        for k, v in sorted(results.items(), key=lambda x: x[1]["wape"])
    ])
    st.dataframe(table_df, hide_index=True, use_container_width=True)

    st.divider()

    # ── Step 3: Future forecast with winner ───────────────────────────────────
    st.subheader(f"🔮 Future Forecast — Next {horizon} Months ({winner})")
    st.caption("Winner retrained on full history, forecasting beyond the data.")

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=df["ds"], y=df["y"],
        name="History", line=dict(color="#4f8ef7", width=2),
    ))
    # Bridge from last actual to first forecast
    bridge_ds = [df["ds"].iloc[-1], forecast_df["ds"].iloc[0]]
    bridge_y = [df["y"].iloc[-1], forecast_df["forecast"].iloc[0]]
    fig_fc.add_trace(go.Scatter(
        x=bridge_ds, y=bridge_y,
        line=dict(color="#f59e0b", width=2, dash="dash"),
        showlegend=False,
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["forecast"],
        name=f"Forecast ({winner})", mode="lines+markers",
        line=dict(color="#f59e0b", width=3, dash="dash"),
        marker=dict(size=8),
    ))
    fig_fc.update_layout(
        height=320, margin=dict(t=20, b=0, l=0, r=0),
        xaxis_title="", yaxis_title="Value",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.divider()

    # ── Step 4: Lesson learned ────────────────────────────────────────────────
    st.subheader("💾 Lesson Saved to Memory")
    l_col, m_col = st.columns([3, 1])
    with l_col:
        st.markdown(f"> {lesson['lesson_text']}")
        st.caption(f"**Stored recommendation:** {lesson['recommendation']}")
    with m_col:
        st.metric("Winner WAPE", f"{lesson['winner_wape']:.1%}")
        st.metric("vs Best Loser", f"−{lesson['improvement_pp']}pp")
    st.success("✅ Lesson saved — the agent will use this on the next similar series.")

else:
    st.info("Configure settings in the sidebar and click **▶ Run Forecast** to start.")
