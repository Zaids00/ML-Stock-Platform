import streamlit as st
import pandas as pd
from ml_backend import run_experiment, DEFAULT_CONFIG

st.set_page_config(page_title="ML Stock Dashboard", layout="wide")
st.title("ML-Driven Stock Selection & Backtesting Platform")

with st.sidebar:
    st.header("Controls")
    epochs = st.slider("Epochs", 5, 100, DEFAULT_CONFIG["epochs"])
    buy_top_n = st.slider("Buy Top N", 1, 20, DEFAULT_CONFIG["buy_top_n"])
    hold_top_n = st.slider("Hold Top N", 1, 25, DEFAULT_CONFIG["hold_top_n"])
    min_prob = st.slider("Min Probability", 0.0, 1.0, float(DEFAULT_CONFIG["min_prob"]), 0.01)
    classification_threshold = st.slider(
        "Classification Threshold", 0.0, 1.0,
        float(DEFAULT_CONFIG["classification_threshold"]), 0.01
    )
    run_button = st.button("Run Model")

if "results" not in st.session_state:
    st.session_state["results"] = None

# if run_button:
#     config = DEFAULT_CONFIG.copy()
#     config["epochs"] = epochs
#     config["buy_top_n"] = buy_top_n
#     config["hold_top_n"] = hold_top_n
#     config["min_prob"] = min_prob
#     config["classification_threshold"] = classification_threshold
#     progress_text = st.sidebar.empty()
#     progress_text.info("Training and backtesting in progress...")

#     with st.spinner("Running experiment..."):
#         st.session_state["results"] = run_experiment(config)

#     progress_text.success("Run complete.")
if run_button:
    config = DEFAULT_CONFIG.copy()
    config["epochs"] = epochs
    config["buy_top_n"] = buy_top_n
    config["hold_top_n"] = hold_top_n
    config["min_prob"] = min_prob
    config["classification_threshold"] = classification_threshold

    progress_bar = st.sidebar.progress(0)
    progress_text = st.sidebar.empty()

    def update_progress(current_epoch, total_epochs):
        percent = int((current_epoch / total_epochs) * 100)
        progress_bar.progress(percent)
        progress_text.info(f"Epoch {current_epoch} / {total_epochs} ({percent}%)")

    with st.spinner("Training model..."):
        st.session_state["results"] = run_experiment(
            config,
            progress_callback=update_progress
        )

    progress_text.success("Training complete.") 

results = st.session_state["results"]

tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Live Signals",
    "Backtesting",
    "Training"
])

if results is not None:
    metrics = results["metrics"]
    top_picks_df = results["top_picks_df"]
    trade_log_df = results["trade_log_df"]
    portfolio_history_df = results["portfolio_history_df"]
    spy_benchmark_df = results["spy_benchmark_df"]
    feature_importance_df = results["feature_importance_df"]
    predictions_df = results["predictions_df"]
    training_history_df = results["training_history_df"]

    with tab1:
        st.subheader("Dashboard")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f'{metrics["test_accuracy"]:.3f}')
        c2.metric("Portfolio Value", f'${metrics["final_portfolio_value"]:.2f}')
        c3.metric("Total Return", f'{metrics["total_return"] * 100:.2f}%')
        c4.metric("Max Drawdown", f'{metrics["max_drawdown"] * 100:.2f}%')

        st.subheader("Equity Curve")
        equity_df = portfolio_history_df[["date", "total_equity"]].copy()
        equity_df = equity_df.merge(
            spy_benchmark_df[["date", "spy_equity"]],
            on="date",
            how="left"
        ).set_index("date")
        st.line_chart(equity_df)

        st.subheader("Top Picks")
        st.dataframe(top_picks_df, use_container_width=True)

    with tab2:
        st.subheader("Live Signals")

        st.markdown("### Buy Now")
        st.dataframe(top_picks_df, use_container_width=True)

        st.markdown("### Current Portfolio Snapshot")
        c1, c2, c3 = st.columns(3)
        c1.metric("Open Positions", metrics["open_positions_end"])
        c2.metric("Buy Trades", metrics["num_buy_trades"])
        c3.metric("Sell Trades", metrics["num_sell_trades"])

        st.markdown("### Recent Actions")
        st.dataframe(trade_log_df.tail(15), use_container_width=True)

    with tab3:
        st.subheader("Backtesting")

        equity_df = portfolio_history_df[["date", "total_equity"]].copy()
        equity_df = equity_df.merge(
            spy_benchmark_df[["date", "spy_equity"]],
            on="date",
            how="left"
        ).set_index("date")
        st.line_chart(equity_df)

        st.markdown("### Trade Log")
        st.dataframe(trade_log_df, use_container_width=True)

        st.markdown("### Portfolio History")
        st.dataframe(portfolio_history_df.tail(30), use_container_width=True)

    with tab4:
        st.subheader("Training")

        final_epoch = int(training_history_df["epoch"].iloc[-1])
        final_loss = float(training_history_df["loss"].iloc[-1])
        final_train_acc = float(training_history_df["train_accuracy"].iloc[-1])

        c1, c2, c3 = st.columns(3)
        c1.metric("Epochs Completed", final_epoch)
        c2.metric("Final Loss", f"{final_loss:.4f}")
        c3.metric("Final Train Accuracy", f"{final_train_acc:.3f}")

        st.markdown("### Training Loss")
        st.line_chart(training_history_df.set_index("epoch")["loss"])

        st.markdown("### Training Accuracy")
        st.line_chart(training_history_df.set_index("epoch")["train_accuracy"])

        st.markdown("### Feature Importance")
        st.dataframe(feature_importance_df.head(20), use_container_width=True)

        st.markdown("### Prediction Samples")
        st.dataframe(predictions_df.tail(30), use_container_width=True)

else:
    st.info("Adjust the settings and click Run Model.")