import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data import DEMO_SERIES
from agents.profiler import profile_series, describe_profile
from agents.tournament import run_tournament, run_final_forecast
from agents.memory import save_lesson, retrieve_similar
from agents.reflection import generate_lesson

st.set_page_config(page_title="MemoryCast", page_icon="🧠", layout="wide")

st.title("🧠 MemoryCast")
st.caption("A self-improving forecasting agent that learns which strategies work best over time.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Forecast Settings")
    series_name = st.selectbox("Select time series", list(DEMO_SERIES.keys()))
    horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=6, value=3)
    run = st.button("▶ Run Forecast", type="primary", use_container_width=True)

    st.divider()
    st.caption("**Demo story:**\n1. Run *Retail Demand* → ETS lesson saved\n2. Run *Support Volume* → ML lesson saved\n3. Run *Cash Inflow* → memory retrieved, agent recommends GradientBoosting")

df = DEMO_SERIES[series_name]
uid = df["unique_id"].iloc[0]

# ── Top row: Series Profile + Memory ─────────────────────────────────────────
col_profile, col_memory = st.columns(2)

with col_profile:
    st.subheader("📊 Series Profile")
    profile = profile_series(df)
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
            diff = abs(sp.get("volatility", 0) - profile["volatility"]) + \
                   abs(sp.get("seasonality_strength", 0) - profile["seasonality_strength"]) + \
                   abs(sp.get("trend_strength", 0) - profile["trend_strength"])
            similarity = max(0, round((1 - diff / 3) * 100))
            source = mem.get("series_label") or "past series"
            model = mem.get("winning_model", "?")
            wape = mem.get("winner_wape", 0)
            with st.expander(f"📌 {source} — **{model}** won · {similarity}% match", expanded=True):
                st.markdown(mem.get("lesson_text", ""))
                st.caption(f"WAPE: {wape:.1%} · {mem.get('saved_at','')[:10]}")

        # Agent synthesis: majority vote across retrieved memories
        from collections import Counter
        winner_counts = Counter(m.get("winning_model") for m in memories)
        top_model, top_count = winner_counts.most_common(1)[0]
        total = len(memories)
        if top_count > 1:
            st.success(f"💡 **{top_count}/{total} similar past runs** favoured **{top_model}** — agent will prioritise this in the tournament")
        else:
            st.warning(f"💡 **Mixed results** across {total} similar past runs — running full tournament to decide")
    else:
        st.info("No prior memory found — running full model tournament for the first time.")

st.divider()

# ── Tournament Results ────────────────────────────────────────────────────────
if run:
    st.subheader("⚔️ Model Tournament")
    with st.spinner("Running backtests across all models…"):
        results, winner = run_tournament(df, horizon=horizon)

    # WAPE bar chart
    model_names = list(results.keys())
    wapes = [results[m]["wape"] * 100 for m in model_names]
    colors = ["#22c55e" if m == winner else "#94a3b8" for m in model_names]

    fig_bar = go.Figure(go.Bar(
        x=model_names, y=wapes,
        marker_color=colors,
        text=[f"{w:.1f}%" for w in wapes],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title=f"WAPE by model — Winner: {winner} 🏆",
        yaxis_title="WAPE %", height=320,
        margin=dict(t=40, b=0, l=0, r=0),
        yaxis=dict(range=[0, max(wapes) * 1.25]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Results table
    results_df = pd.DataFrame(
        [{"Model": k, "WAPE": f"{v['wape']:.1%}", "Bias": f"{v['bias']:+.1%}", "Type": v["type"], "Winner": "🏆" if k == winner else ""}
         for k, v in results.items()]
    ).sort_values("WAPE")
    st.dataframe(results_df, hide_index=True, use_container_width=True)

    # Final forecast
    with st.spinner("Generating forecast with winning model…"):
        forecast_df = run_final_forecast(df, horizon, winner)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="History", line=dict(color="#4f8ef7")))
    fig_fc.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["forecast"], name=f"Forecast ({winner})",
                                 line=dict(color="#f59e0b", dash="dash"), mode="lines+markers"))
    fig_fc.update_layout(title=f"Forecast — {winner}", height=300, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_fc, use_container_width=True)

    st.divider()

    # ── Lesson Learned ────────────────────────────────────────────────────────
    st.subheader("💾 Lesson Learned")
    lesson = generate_lesson(profile, results, winner, series_label=series_name)

    lesson_col, save_col = st.columns([3, 1])
    with lesson_col:
        st.markdown(f"> {lesson['lesson_text']}")
        st.caption(f"**Recommendation stored:** {lesson['recommendation']}")

    with save_col:
        st.metric("Winner WAPE", f"{lesson['winner_wape']:.1%}")
        st.metric("vs Best Loser", f"-{lesson['improvement_pp']}pp")

    save_lesson(lesson)
    st.success("✅ Lesson saved to memory. The agent will use this on the next similar series.")

else:
    st.info("Configure settings in the sidebar and click **▶ Run Forecast** to start.")
