import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Trader Behavior vs Market Sentiment",
    layout="wide"
)

st.title("📊 Trader Behavior vs Fear & Greed Analysis")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    try:
        trades = pd.read_csv("historical_data.csv")
        sentiment = pd.read_csv("fear_greed_index.csv")
    except Exception as e:
        st.error(f"❌ File loading error: {e}")
        return pd.DataFrame()

    # ================= DATE ALIGNMENT =================
    # ⭐ CRITICAL FIX: dayfirst=True
    trades["date"] = pd.to_datetime(
        trades["Timestamp_IST"],
        dayfirst=True,
        errors="coerce"
    ).dt.date

    sentiment["date"] = pd.to_datetime(
        sentiment["timestamp"],
        unit="s",          # ⭐⭐⭐ THIS FIXES 1970 ISSUE
        errors="coerce"
    ).dt.date

    # ================= MERGE =================
    merged = trades.merge(
        sentiment[["date", "value", "classification"]],
        on="date",
        how="left"
    )

    # ================= NUMERIC CLEANING =================
    merged["Closed_PnL"] = pd.to_numeric(
        merged["Closed_PnL"], errors="coerce"
    )

    merged["Size_USD"] = pd.to_numeric(
        merged["Size_USD"], errors="coerce"
    )

    merged["Side"] = (
        merged["Side"].astype(str).str.lower().str.strip()
    )

    merged["win"] = (merged["Closed_PnL"] > 0).astype(int)

    # ================= DEBUG PANEL =================
    st.sidebar.subheader("🧪 Debug Info")
    st.sidebar.write(
        "Trades date range:",
        trades["date"].min(), "→", trades["date"].max()
    )
    st.sidebar.write(
        "Sentiment date range:",
        sentiment["date"].min(), "→", sentiment["date"].max()
    )
    st.sidebar.write("Merged shape:", merged.shape)
    st.sidebar.write(
        "Null classification:",
        merged["classification"].isna().sum()
    )

    return merged


df = load_data()

# ================= SAFETY CHECKS =================
if df.empty:
    st.error("❌ Dataframe is empty after loading.")
    st.stop()

if df["classification"].isna().all():
    st.error("❌ Sentiment merge failed — no matching dates.")
    st.stop()

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Behavior Analysis",
    "🧠 Trader Archetypes",
    "🤖 ML Prediction"
])

# =====================================================
# TAB 1 — OVERVIEW
# =====================================================
with tab1:
    st.subheader("📌 Key Performance Indicators")

    total_trades = len(df)
    total_pnl = df["Closed_PnL"].sum()
    avg_trade = df["Size_USD"].mean()
    win_rate = df["win"].mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Trades", f"{total_trades:,}")
    k2.metric("Total PnL", f"{total_pnl:,.2f}")
    k3.metric("Avg Trade Size", f"{avg_trade:,.2f}")
    k4.metric("Win Rate", f"{win_rate:.2f}%")

    st.divider()

    # ---------- PERFORMANCE BY SENTIMENT ----------
    st.subheader("📊 Performance by Sentiment")

    perf_df = df.dropna(subset=["classification", "Closed_PnL"])

    if perf_df.empty:
        st.warning("No data available after cleaning.")
    else:
        perf_sent = (
            perf_df.groupby("classification")
            .agg(
                Avg_PnL=("Closed_PnL", "mean"),
                Win_Rate=("win", "mean"),
                Trades=("Closed_PnL", "count"),
            )
            .reset_index()
        )

        perf_sent["Win_Rate"] *= 100

        colA, colB = st.columns(2)

        with colA:
            fig1 = px.bar(
                perf_sent,
                x="classification",
                y="Avg_PnL",
                title="Average PnL by Sentiment",
            )
            st.plotly_chart(fig1, use_container_width=True)

        with colB:
            fig2 = px.bar(
                perf_sent,
                x="classification",
                y="Win_Rate",
                title="Win Rate (%) by Sentiment",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(perf_sent, use_container_width=True)

# =====================================================
# TAB 2 — BEHAVIOR
# =====================================================
with tab2:
    st.subheader("📈 Trader Behavior")

    col1, col2 = st.columns(2)

    daily_trades = (
        df.groupby("date")
        .size()
        .reset_index(name="trades")
    )

    with col1:
        fig3 = px.line(
            daily_trades,
            x="date",
            y="trades",
            title="Daily Trade Frequency"
        )
        st.plotly_chart(fig3, use_container_width=True)

    side_dist = df["Side"].value_counts().reset_index()
    side_dist.columns = ["Side", "Count"]

    with col2:
        fig4 = px.pie(
            side_dist,
            names="Side",
            values="Count",
            title="Buy vs Sell Ratio"
        )
        st.plotly_chart(fig4, use_container_width=True)

# =====================================================
# TAB 3 — ARCHETYPES
# =====================================================
with tab3:
    st.subheader("🧠 Trader Behavioral Archetypes")

    arch_df = df.dropna(subset=["Size_USD", "Closed_PnL"]).copy()

    trader_stats = (
        arch_df.groupby("Account")
        .agg(
            total_trades=("Account", "count"),
            avg_size=("Size_USD", "mean"),
            total_pnl=("Closed_PnL", "sum"),
            win_rate=("win", "mean")
        )
        .reset_index()
    ).fillna(0)

    size_med = trader_stats["avg_size"].median()
    trade_med = trader_stats["total_trades"].median()
    win_med = trader_stats["win_rate"].median()

    def classify_trader(row):
        if row["avg_size"] > size_med * 1.5 and row["win_rate"] < win_med:
            return "High Leverage Gambler"
        elif row["win_rate"] > 0.6 and row["total_trades"] > trade_med:
            return "Consistent Winner"
        elif row["total_trades"] > trade_med * 1.5 and row["win_rate"] < win_med:
            return "Overtrader"
        elif row["win_rate"] > win_med and row["total_pnl"] > 0:
            return "Smart Money"
        else:
            return "Casual Trader"

    trader_stats["Archetype"] = trader_stats.apply(classify_trader, axis=1)

    arch_dist = trader_stats["Archetype"].value_counts().reset_index()
    arch_dist.columns = ["Archetype", "Count"]

    fig5 = px.bar(
        arch_dist,
        x="Archetype",
        y="Count",
        title="Archetype Distribution"
    )
    st.plotly_chart(fig5, use_container_width=True)

# =====================================================
# TAB 4 — ML
# =====================================================
with tab4:
    st.subheader("🤖 Predict Trader Win Probability")

    ml_df = df.dropna(subset=["Size_USD", "Closed_PnL"]).copy()

    features = ["Size_USD", "value"]
    X = ml_df[features].fillna(0)
    y = ml_df["win"].astype(int)

    if len(X) > 100:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1  # ⭐ prevents Windows warning
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.metric("Model Accuracy", f"{acc:.2f}")

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        })

        fig_imp = px.bar(
            importance,
            x="Feature",
            y="Importance",
            title="Feature Importance"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Not enough data for ML model.")