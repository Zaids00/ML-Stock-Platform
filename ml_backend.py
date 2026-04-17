import yfinance as yf
import pandas as pd
import tensorflow as tf


def clean_yf_download(symbol, period, interval):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return df

    # Flatten MultiIndex columns safely
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Prefer selecting this symbol from the second level
            if symbol in df.columns.get_level_values(-1):
                df = df.xs(symbol, axis=1, level=-1)
            else:
                df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = df.columns.get_level_values(0)

    # If duplicate columns still exist, keep the first copy
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Ensure required OHLCV columns are plain Series
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{symbol}: missing required column '{col}'")

        # If somehow still duplicated / dataframe-like, collapse to first column
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    return df

DEFAULT_CONFIG = {
    "epochs": 60,
    "learning_rate": 0.001, 
    "require_positive_return": True,
    "buy_top_n": 5,
    "hold_top_n": 15,
    "min_prob": 0.52,
    "classification_threshold": 0.6,
    "buy_threshold": 0.6,
    "min_expected_return": 0.0,
    "max_positions": 15,
    "fee_per_trade": 0.001,
    "initial_capital": 1000,
    "lookback_years": 10,
    "period": "10y",
    "interval": "1d",
    "batch_size": 64,
    "learning_rate": 0.001,
    "tickers": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
        "TSLA", "AMD", "NFLX", "INTC", "CSCO", "ADBE",
        "CRM", "QCOM", "AVGO", "TXN", "ORCL", "IBM",
        "JPM", "GS", "MS", "BAC", "WFC",
        "XOM", "CVX",
        "JNJ", "PFE", "UNH", "MRK",
        "KO", "PEP", "WMT", "COST", "MCD",
        "DIS", "NKE", "SBUX",
        "CAT", "BA", "GE", "HON"
    ]
}

class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_callback=None):
        super().__init__()
        self.progress_callback = progress_callback

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_callback is not None:
            self.progress_callback(epoch + 1, self.params["epochs"])


def run_experiment(config, progress_callback=None):
    tickers = config["tickers"]
    lookback_years = config.get("lookback_years", 10)
    period = f"{lookback_years}y"
    interval = config["interval"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    classification_threshold = config["classification_threshold"]
    min_expected_return = config.get("min_expected_return", 0.0)
    buy_top_n = config["buy_top_n"]
    hold_top_n = config["hold_top_n"]
    max_positions = config["max_positions"]
    fee_per_trade = config["fee_per_trade"]
    min_prob = config["min_prob"]
    initial_capital = config["initial_capital"]
    ETF_LIST = ["SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]
    asset_universe_mode = config.get("asset_universe_mode", "Stocks + ETFs")

    if asset_universe_mode == "Stocks Only":
        tickers = [t for t in tickers if t not in ETF_LIST]

    # tickers = [
    #     "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    #     "TSLA", "AMD", "NFLX", "INTC", "CSCO", "ADBE",
    #     "CRM", "QCOM", "AVGO", "TXN", "ORCL", "IBM",
    #     "JPM", "GS", "MS", "BAC", "WFC",
    #     "XOM", "CVX",
    #     "JNJ", "PFE", "UNH", "MRK",
    #     "KO", "PEP", "WMT", "COST", "MCD",
    #     "DIS", "NKE", "SBUX",
    #     "CAT", "BA", "GE", "HON"
    # ]

    all_data = []

    # spy = yf.download("SPY", period=period, interval=interval, auto_adjust=False)

    # if isinstance(spy.columns, pd.MultiIndex):
    #     spy.columns = spy.columns.get_level_values(0)
    spy = clean_yf_download("SPY", period, interval)

    spy["spy_return_1d"] = spy["Close"].pct_change(1)
    spy["spy_return_3d"] = spy["Close"].pct_change(3)
    spy = spy[["spy_return_1d", "spy_return_3d"]]

    sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]

    sector_data = {}

    for etf in sector_etfs:
        # etf_df = yf.download(etf, period=period, interval=interval, auto_adjust=False)
        # if isinstance(etf_df.columns, pd.MultiIndex):
        #     etf_df.columns = etf_df.columns.get_level_values(0)
        etf_df = clean_yf_download(etf, period, interval)

        etf_df[f"{etf}_return_1d"] = etf_df["Close"].pct_change(1)
        etf_df[f"{etf}_return_3d"] = etf_df["Close"].pct_change(3)

        sector_data[etf] = etf_df[[f"{etf}_return_1d", f"{etf}_return_3d"]]

    for ticker in tickers:
        # data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

        # if isinstance(data.columns, pd.MultiIndex): 
        #     data.columns = data.columns.get_level_values(0)
        data = clean_yf_download(ticker, period, interval)
        data["Ticker"] = ticker

        data = data.merge(spy, left_index=True, right_index=True, how="left")

        for etf in sector_etfs:
            data = data.merge(sector_data[etf], left_index=True, right_index=True, how="left")


        # ----------------------------
        # 2. 1-day future return target
        # ----------------------------
        future_return_1d = data["Close"].shift(-1) / data["Close"] - 1
        data["target"] = (future_return_1d > 0).astype(int)
        data["future_return_1d"] = future_return_1d
        data["target_return"] = future_return_1d

        # ----------------------------
        # 3. Feature engineering
        # ----------------------------
        data["return_1d"] = data["Close"].pct_change(1)
        data["return_3d"] = data["Close"].pct_change(3)
        data["return_5d"] = data["Close"].pct_change(5)
        data["return_10d"] = data["Close"].pct_change(10)
        data["return_20d"] = data["Close"].pct_change(20)
        data["return_30d"] = data["Close"].pct_change(30)
        data["return_60d"] = data["Close"].pct_change(60)

        data["ma_5"] = data["Close"].rolling(5).mean()
        data["ma_10"] = data["Close"].rolling(10).mean()
        data["ma_20"] = data["Close"].rolling(20).mean()

        # Price vs MA ratios
        data["ma_ratio_5"] = data["Close"] / data["ma_5"]
        data["ma_ratio_10"] = data["Close"] / data["ma_10"]
        data["ma_ratio_20"] = data["Close"] / data["ma_20"]

        # MA cross ratios
        data["ma_ratio_5_20"] = data["ma_5"] / data["ma_20"]
        data["ma_ratio_10_20"] = data["ma_10"] / data["ma_20"]

        # Volatility
        data["volatility_1"] = data["Close"].pct_change().rolling(1).std()
        data["volatility_3"] = data["Close"].pct_change().rolling(3).std()
        data["volatility_5"] = data["Close"].pct_change().rolling(5).std()
        data["volatility_10"] = data["Close"].pct_change().rolling(10).std()
        data["volatility_20"] = data["Close"].pct_change().rolling(20).std()
        data["volatility_30"] = data["Close"].pct_change().rolling(30).std()
        data["volatility_60"] = data["Close"].pct_change().rolling(60).std()


        # Volume features
        data["volume_ratio_10"] = data["Volume"] / data["Volume"].rolling(10).mean()
        data["volume_ratio_20"] = data["Volume"] / data["Volume"].rolling(20).mean()
        data["volume_ratio_30"] = data["Volume"] / data["Volume"].rolling(30).mean()

        # Candle / range features
        data["high_low_range"] = (data["High"] - data["Low"]) / data["Close"]
        data["close_open_ratio"] = (data["Close"] - data["Open"]) / data["Open"]

        data["high_252"] = data["Close"].rolling(252).max()
        data["dist_from_52w_high"] = data["Close"] / data["high_252"]

        # RSI
        delta = data["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        data["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = data["Close"].ewm(span=12, adjust=False).mean()
        ema26 = data["Close"].ewm(span=26, adjust=False).mean()
        data["macd"] = ema12 - ema26
        

        # Relative strength vs SPY
        data["relative_strength_1d"] = data["return_1d"] - data["spy_return_1d"]
        data["relative_strength_3d"] = data["return_3d"] - data["spy_return_3d"]

        # Momentum Accelaration
        data["momentum_accelaration"] = data["return_5d"] - data["return_20d"]
        data["minimomentum_accelaration"] = data["return_1d"] - data["return_3d"]



        data = data.dropna(subset=["target"])
        data = data.ffill()
        data = data.fillna(0)
        all_data.append(data)
        print("total rows after cleanup",len(data))


    data = pd.concat(all_data).sort_index()

    # ----------------------------
    # Cross-sectional momentum ranking
    # ----------------------------
    data["rank_return_60d"] = (
        data.groupby(data.index)["return_60d"]
        .rank(pct=True)
    )
    data["rank_return_30d"] = (
        data.groupby(data.index)["return_30d"]
        .rank(pct=True)
    )
    data["rank_return_20d"] = (
        data.groupby(data.index)["return_20d"]
        .rank(pct=True)
    )
    data["rank_return_10d"] = (
        data.groupby(data.index)["return_10d"]
        .rank(pct=True)
    )
    data["rank_return_5d"] = (
        data.groupby(data.index)["return_5d"]
        .rank(pct=True)
    )
    data["rank_return_3d"] = (
        data.groupby(data.index)["return_3d"]
        .rank(pct=True)
    )
    data["rank_return_1d"] = (
        data.groupby(data.index)["return_1d"]
        .rank(pct=True)
    )
    data["rank_volatility_1d"] = (
        data.groupby(data.index)["volatility_1"]
        .rank(pct=True)
    )
    data["rank_volatility_3d"] = (
        data.groupby(data.index)["volatility_3"]
        .rank(pct=True)
    )
    data["rank_volatility_5d"] = (
        data.groupby(data.index)["volatility_5"]
        .rank(pct=True)
    )
    data["rank_volatility_10d"] = (
        data.groupby(data.index)["volatility_10"]
        .rank(pct=True)
    )
    data["rank_volatility_20d"] = (
        data.groupby(data.index)["volatility_20"]
        .rank(pct=True)
    )
    data["rank_volatility_30d"] = (
        data.groupby(data.index)["volatility_30"]
        .rank(pct=True)
    )
    data["rank_volatility_60d"] = (
        data.groupby(data.index)["volatility_60"]
        .rank(pct=True)
    )


    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "return_60d",
        "rank_return_60d", "rank_return_30d", "rank_return_20d", "rank_return_10d", "rank_return_5d", "rank_return_3d", "rank_return_1d", 
        "ma_5", "ma_10", "ma_20",
        "ma_ratio_5", "ma_ratio_10", "ma_ratio_20",
        "ma_ratio_5_20", "ma_ratio_10_20",
        "volatility_20", "volatility_3", "volatility_60",
        "rank_volatility_3d", "rank_volatility_20d","rank_volatility_60d",
        "volume_ratio_10", "volume_ratio_20", "volume_ratio_30",
        "high_low_range", "close_open_ratio", "dist_from_52w_high",
        "rsi", "macd", "minimomentum_accelaration", "momentum_accelaration",
        "spy_return_1d", "spy_return_3d",
        "relative_strength_1d", "relative_strength_3d",
        "XLK_return_1d", "XLK_return_3d",
        "XLF_return_1d", "XLF_return_3d",
        "XLE_return_1d", "XLE_return_3d",
        "XLV_return_1d", "XLV_return_3d",
        "XLY_return_1d", "XLY_return_3d",
        "XLI_return_1d", "XLI_return_3d",
    ]

    print("Feature count:", len(feature_cols))
    train_parts = []
    test_parts = []

    for ticker in tickers:
        stock_data = data[data["Ticker"] == ticker].copy()
        split = int(0.8 * len(stock_data))
        train_parts.append(stock_data.iloc[:split])
        test_parts.append(stock_data.iloc[split:])

    train_data = pd.concat(train_parts)
    test_data = pd.concat(test_parts)

    # X_train_df = train_data[feature_cols].copy()
    # y_train_df = train_data["target"].copy()

    # X_test_df = test_data[feature_cols].copy()
    # y_test_df = test_data["target"].copy()
    X_train_df = train_data[feature_cols].copy()
    y_train_cls_df = train_data["target"].copy()
    y_train_ret_df = train_data["target_return"].copy()

    X_test_df = test_data[feature_cols].copy()
    y_test_cls_df = test_data["target"].copy()
    y_test_ret_df = test_data["target_return"].copy()

    print("Train rows:", len(X_train_df))
    print("Test rows:", len(X_test_df))

    # ----------------------------
    # 6. Normalize using TRAIN only
    # ----------------------------
    feature_means = X_train_df.mean()
    feature_stds = X_train_df.std().replace(0, 1)

    X_train_df = (X_train_df - feature_means) / feature_stds
    X_test_df = (X_test_df - feature_means) / feature_stds

    X_train_df = X_train_df.fillna(0)
    X_test_df = X_test_df.fillna(0)

    # X_train = tf.convert_to_tensor(X_train_df.values, dtype=tf.float32)
    # y_train = tf.convert_to_tensor(y_train_df.values, dtype=tf.float32)

    # X_test = tf.convert_to_tensor(X_test_df.values, dtype=tf.float32)
    # y_test = tf.convert_to_tensor(y_test_df.values, dtype=tf.float32)
    X_train = tf.convert_to_tensor(X_train_df.values, dtype=tf.float32)
    y_train_cls = tf.convert_to_tensor(y_train_cls_df.values, dtype=tf.float32)
    y_train_ret = tf.convert_to_tensor(y_train_ret_df.values, dtype=tf.float32)

    X_test = tf.convert_to_tensor(X_test_df.values, dtype=tf.float32)
    y_test_cls = tf.convert_to_tensor(y_test_cls_df.values, dtype=tf.float32)
    y_test_ret = tf.convert_to_tensor(y_test_ret_df.values, dtype=tf.float32)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # ----------------------------
    # 7. Baseline accuracy
    # ----------------------------
    # ----------------------------
    # 7. Baseline accuracy
    # ----------------------------
    baseline_class = 1.0 if float(tf.reduce_mean(y_train_cls).numpy()) >= 0.5 else 0.0
    baseline_acc = tf.reduce_mean(
        tf.cast(tf.equal(y_test_cls, baseline_class), tf.float32)
    )

    print("\nBaseline test accuracy:", baseline_acc.numpy())

    # ----------------------------
    # 9. Model
    # ----------------------------
    inputs = tf.keras.Input(shape=(X_train.shape[1],), name="features")
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    cls_out = tf.keras.layers.Dense(1, name="cls_out")(x)
    ret_out = tf.keras.layers.Dense(1, name="ret_out")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={"cls_out": cls_out, "ret_out": ret_out}
    )

    learning_rate = config.get("learning_rate", 0.001)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        # optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "cls_out": tf.keras.losses.BinaryCrossentropy(from_logits=True),
            "ret_out": tf.keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "cls_out": 1.0,
            "ret_out": 0.5,
        },
        metrics={
            "cls_out": ["accuracy"],
            "ret_out": [tf.keras.metrics.MeanSquaredError()],
        }
    )

    progress_cb = StreamlitProgressCallback(progress_callback)

    history = model.fit(
        X_train,
        {"cls_out": y_train_cls, "ret_out": y_train_ret},
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[progress_cb]
    )

    training_history = []
    for epoch in range(epochs):
        training_history.append({
            "epoch": epoch + 1,
            "loss": float(history.history["loss"][epoch]),
            "train_accuracy": float(history.history["cls_out_accuracy"][epoch])
        })

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}: "
                f"Loss = {history.history['loss'][epoch]:.6f}, "
                f"Train Accuracy = {history.history['cls_out_accuracy'][epoch]:.4f}"
            )



    # ----------------------------
    # 10. Test evaluation
    # ----------------------------
    # ----------------------------
    # 10. Test evaluation
    # ----------------------------
    preds = model.predict(X_test, verbose=0)

    test_logits = tf.squeeze(preds["cls_out"])
    predicted_return = tf.squeeze(preds["ret_out"]).numpy()

    test_probs = tf.sigmoid(test_logits)
    test_preds = tf.cast(test_probs > classification_threshold, tf.float32)

    test_acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, y_test_cls), tf.float32))

    print("\nTest Accuracy:", test_acc.numpy())

    print("\nFirst 10 test probabilities:")
    print(test_probs[:10].numpy())

    print("\nFirst 10 predicted returns:")
    print(predicted_return[:10])

    print("\nFirst 10 predictions vs actual:")
    for i in range(10):
        print(
            "Prob Up:", float(test_probs[i].numpy()),
            "Pred:", int(test_preds[i].numpy()),
            "Actual:", int(y_test_cls[i].numpy()),
            "Pred Return:", float(predicted_return[i]),
            "Actual Return:", float(y_test_ret[i].numpy())
        )

    print("\nMean predicted probability:", float(tf.reduce_mean(test_probs).numpy()))
    print("Fraction predicted UP:", float(tf.reduce_mean(tf.cast(test_preds, tf.float32)).numpy()))
    print("Actual fraction UP:", float(tf.reduce_mean(y_test_cls).numpy()))

    # ----------------------------
    # Portfolio backtest: long only, with sell/rotation
    # ----------------------------
    # initial_capital = 1000.0
    capital = initial_capital

    # buy_threshold = 0.65    
    # sell_threshold = 0.50
    # max_positions = 10
    # fee_per_trade = 0.001

    # buy_top_n = 10
    # hold_top_n = 15   
    # max_positions = 10
    # fee_per_trade = 0.001

    test_data = test_data.copy()
    test_data["prob_up"] = test_probs.numpy()
    test_data["predicted_return"] = predicted_return
    test_data["predicted_price"] = test_data["Close"] * (1 + test_data["predicted_return"])

    test_data["pred"] = (test_data["prob_up"] > classification_threshold).astype(int)

    test_data["rank"] = (
    test_data.groupby(test_data.index)["prob_up"]
    .rank(ascending=False, method="first")
    .astype(int)
    )
    
    predictions_df = test_data.reset_index().copy()

    if "Date" in predictions_df.columns:
        predictions_df = predictions_df.rename(columns={"Date": "date"})
    elif "index" in predictions_df.columns:
        predictions_df = predictions_df.rename(columns={"index": "date"})

    predictions_df = predictions_df[[
        "date", "Ticker", "Open", "High", "Low", "Close",
        "prob_up", "pred", "target", "future_return_1d",
        "volatility_20", "predicted_return", "predicted_price", "rank"
    ]]

    latest_date = predictions_df["date"].max()

    top_picks_df = (
        predictions_df[predictions_df["date"] == latest_date]
        .sort_values("prob_up", ascending=False)
        .head(buy_top_n)
        .copy()
    )

    top_picks_df["rank"] = range(1, len(top_picks_df) + 1)
    # We'll process one trading day at a time
    test_data = test_data.sort_index()

    portfolio = {}  # ticker -> {"value": float}
    equity_curve = [capital]
    trade_log = []
    # min_prob = 0.52

    portfolio_history = [{
        "date": None,
        "cash": capital,
        "holdings_value": 0.0,
        "total_equity": capital
    }]
    unique_dates = sorted(test_data.index.unique())

    for current_date in unique_dates:
        day_data = test_data.loc[current_date]

        # If only one stock exists that day, convert to DataFrame
        if isinstance(day_data, pd.Series):
            day_data = day_data.to_frame().T
            
        ranking_method = config.get("ranking_method", "probability")

        if ranking_method == "probability":
            day_data = day_data.sort_values("prob_up", ascending=False)

        elif ranking_method == "return":
            day_data = day_data.sort_values("predicted_return", ascending=False)

        elif ranking_method == "combined":
            day_data["combined_score"] = (
                0.6 * day_data["prob_up"] +
                0.4 * day_data["predicted_return"]
            )
            day_data = day_data.sort_values("combined_score", ascending=False)

        day_data = day_data.copy()
        day_data["rank"] = range(1, len(day_data) + 1)  

        # 1. Update existing holdings using today's realized 3-day return proxy
        #    Since your target is 3-day return, this is still a simplified approximation.
        tickers_to_remove = []

        for ticker in list(portfolio.keys()):
            row = day_data[day_data["Ticker"] == ticker]

            if len(row) == 0:
                continue

            actual_return = float(row["future_return_1d"].iloc[0])

            # update holding value
            portfolio[ticker]["value"] *= (1 + actual_return)

            # sell rule: signal has weakened
            # prob_up = float(row["prob_up"].iloc[0])
            # if prob_up < sell_threshold:
            #     sell_value = portfolio[ticker]["value"] * (1 - fee_per_trade)
            #     capital += sell_value

            #     trade_log.append({
            #         "date": current_date,
            #         "ticker": ticker,
            #         "action": "SELL",
            #         "prob_up": prob_up,
            #         "value": sell_value
            #     })

            #     tickers_to_remove.append(ticker)
            prob_up = float(row["prob_up"].iloc[0])
            rank = int(row["rank"].iloc[0])

            if rank > hold_top_n:
                entry_value = portfolio[ticker]["entry_value"]
                sell_value = portfolio[ticker]["value"] * (1 - fee_per_trade)
                profit = sell_value - entry_value
                capital += sell_value
                profit_pct = ((profit / entry_value) * 100) if entry_value != 0 else 0

                trade_log.append({
                    "date": current_date,
                    "ticker": ticker,
                    "action": "SELL",
                    "rank": rank,
                    "prob_up": prob_up,
                    "value": sell_value,
                    "profit": profit,
                    "profit_pct": profit_pct
                })

                tickers_to_remove.append(ticker)

        for ticker in tickers_to_remove:
            del portfolio[ticker]

        # # 2. Find buy candidates for today
        # candidates = day_data.sort_values("prob_up", ascending=False).copy()
        # candidates = candidates[candidates["prob_up"] > buy_threshold]
        # candidates = day_data.head(buy_top_n).copy()
        if config.get("require_positive_return", True):
            return_filter = day_data["predicted_return"] > 0
            return_filter = day_data["predicted_return"] > min_expected_return
            
        else:
            return_filter = day_data["predicted_return"] > min_expected_return

        candidates = day_data[
            (~day_data["Ticker"].isin(portfolio.keys())) &
            (day_data["prob_up"] > min_prob) &
            return_filter
        ]


        # 3. Buy until portfolio is full
        available_slots = max_positions - len(portfolio)

        if available_slots > 0 and len(candidates) > 0:
            buy_candidates = candidates[~candidates["Ticker"].isin(portfolio.keys())].head(available_slots)

            if len(buy_candidates) > 0:
                allocation_per_position = capital / max(len(buy_candidates), 1)

                # for _, row in buy_candidates.iterrows():
                #     ticker = row["Ticker"]
                #     prob_up = float(row["prob_up"])

                #     buy_cost = allocation_per_position * fee_per_trade
                #     invested_amount = allocation_per_position - buy_cost

                #     if invested_amount > 0 and capital >= allocation_per_position:
                #         capital -= allocation_per_position
                #         portfolio[ticker] = {"value": invested_amount}

                #         trade_log.append({
                #             "date": current_date,
                #             "ticker": ticker,
                #             "action": "BUY",
                #             "prob_up": prob_up,
                #             "value": invested_amount
                #         })
                for _, row in buy_candidates.iterrows():
                    ticker = row["Ticker"]
                    prob_up = float(row["prob_up"])
                    rank = int(row["rank"])

                    buy_cost = allocation_per_position * fee_per_trade
                    invested_amount = allocation_per_position - buy_cost

                    if invested_amount > 0 and capital >= allocation_per_position:
                        capital -= allocation_per_position
                        portfolio[ticker] = {
                            "value": invested_amount,
                            "entry_value": invested_amount
                        }

                        trade_log.append({
                            "date": current_date,
                            "ticker": ticker,
                            "action": "BUY",
                            "rank": rank,
                            "prob_up": prob_up,
                            "value": invested_amount,
                            "profit": None,
                            "profit_pct": None
                        })
        # 4. Compute total equity = cash + holdings
        holdings_value = sum(pos["value"] for pos in portfolio.values())
        total_equity = capital + holdings_value
        equity_curve.append(total_equity)

        portfolio_history.append({
            "date": current_date,
            "cash": capital,
            "holdings_value": holdings_value,
            "total_equity": total_equity
        })

    trade_log_df = pd.DataFrame(trade_log)

    portfolio_history_df = pd.DataFrame(portfolio_history)
    portfolio_history_df = portfolio_history_df.dropna(subset=["date"]).copy()

    portfolio_history_df["peak_equity"] = portfolio_history_df["total_equity"].cummax()
    portfolio_history_df["drawdown"] = (
        portfolio_history_df["peak_equity"] - portfolio_history_df["total_equity"]
    ) / portfolio_history_df["peak_equity"]

    final_portfolio_value = float(portfolio_history_df["total_equity"].iloc[-1])
    print("\nFinal portfolio value:", final_portfolio_value)

    # spy_for_benchmark = yf.download("SPY", period=period, interval=interval, auto_adjust=False)

    # if isinstance(spy_for_benchmark.columns, pd.MultiIndex):
    #     spy_for_benchmark.columns = spy_for_benchmark.columns.get_level_values(0)
    spy_for_benchmark = clean_yf_download("SPY", period, interval)

    spy_for_benchmark = spy_for_benchmark.loc[portfolio_history_df["date"].min():portfolio_history_df["date"].max()].copy()
    spy_for_benchmark["spy_return"] = spy_for_benchmark["Close"].pct_change().fillna(0)
    spy_for_benchmark["spy_equity"] = initial_capital * (1 + spy_for_benchmark["spy_return"]).cumprod()

    spy_benchmark_df = spy_for_benchmark.reset_index().rename(columns={"Date": "date"})[["date", "Close", "spy_equity"]]

    print(spy_benchmark_df.head())

    # Basic stats
    buy_trades = [t for t in trade_log if t["action"] == "BUY"]
    sell_trades = [t for t in trade_log if t["action"] == "SELL"]

    print("Number of BUY trades:", len(buy_trades))
    print("Number of SELL trades:", len(sell_trades))
    print("Open positions at end:", len(portfolio))

    # Max drawdown
    max_drawdown = float(portfolio_history_df["drawdown"].max())
    print("Max drawdown:", max_drawdown)


    # ----------------------------
    # 12. Accuracy on traded signals only
    # ----------------------------

    weights = model.layers[1].get_weights()[0]  # first Dense layer weights  # shape: (num_features, hidden_units)
    importance = abs(weights).mean(axis=1)

    feature_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)
    
    feature_importance_df = feature_importance_df.sort_values("importance", ascending=False)
    feature_importance_df.insert(0, "Rank", range(1, len(feature_importance_df) + 1))
    
    training_history_df = pd.DataFrame(training_history)
    print(feature_importance_df.head(30))  
    total_return = (final_portfolio_value / initial_capital) - 1
    holdings_rows = []
    for ticker, pos in portfolio.items():
        latest_row = test_data[test_data["Ticker"] == ticker].sort_index().tail(1)

        if len(latest_row) > 0:
            latest_prob = float(latest_row["prob_up"].iloc[0])
            latest_close = float(latest_row["Close"].iloc[0])
            latest_rank = int(latest_row["rank"].iloc[0]) if "rank" in latest_row.columns else None
        else:
            latest_prob = None
            latest_close = None
            latest_rank = None

        holdings_rows.append({
            "Ticker": ticker,
            "position_value": float(pos["value"]),
            "latest_close": latest_close,
            "latest_prob_up": latest_prob,
            "latest_rank": latest_rank,
            "portfolio_weight": float(pos["value"] / final_portfolio_value) if final_portfolio_value > 0 else 0.0
        })

    holdings_df = pd.DataFrame(holdings_rows).sort_values("position_value", ascending=False)
    metrics = {
        "baseline_test_accuracy": float(baseline_acc.numpy()),
        "test_accuracy": float(test_acc.numpy()),
        "final_portfolio_value": final_portfolio_value,
        "max_drawdown": max_drawdown,
        "num_buy_trades": int(len(buy_trades)),
        "num_sell_trades": int(len(sell_trades)),
        "open_positions_end": int(len(portfolio)),
        "mean_predicted_probability": float(tf.reduce_mean(test_probs).numpy()),
        "fraction_predicted_up": float(tf.reduce_mean(tf.cast(test_preds, tf.float32)).numpy()),
        "actual_fraction_up": float(tf.reduce_mean(y_test_cls).numpy()),
        "total_return": total_return,
    }

    # print(predictions_df.head())
    # print(top_picks_df.head())
    # print(trade_log_df.head())
    # print(portfolio_history_df.head())
    # print(metrics)
    return {
        "metrics": metrics,
        "predictions_df": predictions_df,
        "top_picks_df": top_picks_df,
        "trade_log_df": trade_log_df,
        "portfolio_history_df": portfolio_history_df,
        "feature_importance_df": feature_importance_df,
        "spy_benchmark_df": spy_benchmark_df,
        "training_history_df": training_history_df,
        "holdings_df": holdings_df,
    }

if __name__ == "__main__":
    results = run_experiment(DEFAULT_CONFIG)
    print(results["metrics"])
    print(results["top_picks_df"].head())