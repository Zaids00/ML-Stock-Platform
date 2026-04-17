import streamlit as st
import pandas as pd
from ml_backend import run_experiment, DEFAULT_CONFIG
from model_save import save_run, load_default_run, list_saved_runs, load_run
import altair as alt
import plotly.graph_objects as go
import plotly.express as px



st.markdown("""
<style>
[data-testid="stSidebarNav"] {
    display: none;
}
button[title="Settings"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

ALL_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "INTC",
    "CSCO", "ADBE", "CRM", "QCOM", "AVGO", "TXN", "ORCL", "IBM", "MU", "AMAT",
    "LRCX", "KLAC", "PANW", "SNOW", "PLTR", "UBER", "SHOP", "SQ", "PYPL", "INTU",

    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "KKR",

    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX",

    "JNJ", "PFE", "UNH", "MRK", "ABBV", "LLY", "TMO", "DHR", "BMY", "AMGN",

    "KO", "PEP", "WMT", "COST", "PG", "MCD", "SBUX", "NKE", "DIS", "HD",
    "LOW", "TGT", "CMG", "MAR", "BKNG", "ABNB",

    "CAT", "BA", "GE", "HON", "DE", "UPS", "FDX", "RTX", "LMT", "MMM", 

    "SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI"
]

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toggleSpikelines"
    ]
}


if "pending_training_start" not in st.session_state:
    st.session_state["pending_training_start"] = False

def get_change_class(current, baseline):
    if pd.isna(current) or pd.isna(baseline):
        return ""
    if current > baseline:
        return "green-text"
    elif current < baseline:
        return "red-text"
    return "amber-text"


def metric_card(label, value, css_class=""):
    st.markdown(
        f"""
        <div class="panel">
            <div class="subtle" style="font-size:0.95rem; margin-bottom:8px;">{label}</div>
            <div class="{css_class}" style="font-size:2.2rem; font-weight:700;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def highlight_positive_negative(val):
    if pd.isna(val):
        return ""
    elif val > 0:
        return "color: #22c55e; font-weight: 600;"
    elif val < 0:
        return "color: #ef4444; font-weight: 600;"
    elif val == 0:
        return "color: #f59e0b; font-weight: 600;"
    return ""


def highlight_price_change_from_prev(series):
    styles = []
    prev = None

    for val in series:
        if pd.isna(val) or prev is None:
            styles.append("")
        elif val > prev:
            styles.append("color: #22c55e; font-weight: 600;")
        elif val < prev:
            styles.append("color: #ef4444; font-weight: 600;")
        else:
            styles.append("color: #f59e0b; font-weight: 600;")
        prev = val

    return styles

def highlight_action(val):
    if val == "BUY":
        return "color: #22c55e; font-weight: 600;"
    elif val == "SELL":
        return "color: #ef4444; font-weight: 600;"
    return ""

def highlight_profit(val):
    if pd.isna(val):
        return ""
    elif val > 0:
        return "color: #22c55e; font-weight: 600;"
    elif val < 0:
        return "color: #ef4444; font-weight: 600;"
    return ""


def header_with_help(title, help_text):
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 6px;">
        <h3 style="margin: 0;">{title}</h3>
        <span title="{help_text}"
              style="
                cursor: pointer;
                border-radius: 50%;
                border: 1px solid #64748b;
                width: 16px;
                height: 16px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 11px;
                color: #94a3b8;
              ">
            ?
        </span>
    </div>
    """, unsafe_allow_html=True)

def slider_with_input(label, min_val, max_val, default, step, key, disabled=False, help=None):
    slider_key = f"{key}_slider"
    input_key = f"{key}_input"

    if key not in st.session_state:
        st.session_state[key] = default
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state[key]
    if input_key not in st.session_state:
        st.session_state[input_key] = st.session_state[key]

    def sync_from_slider():
        st.session_state[key] = st.session_state[slider_key]
        st.session_state[input_key] = st.session_state[slider_key]

    def sync_from_input():
        st.session_state[key] = st.session_state[input_key]
        st.session_state[slider_key] = st.session_state[input_key]

    col1, col2 = st.columns([4, 1.4], vertical_alignment="bottom")

    with col1:
        st.slider(
            label,
            min_val,
            max_val,
            step=step,
            key=slider_key,
            disabled=disabled,
            help=help,
            label_visibility="visible",
            on_change=sync_from_slider
        )

    with col2:
        st.number_input(
            "Value",
            min_value=min_val,
            max_value=max_val,
            step=step,
            key=input_key,
            disabled=disabled,
            label_visibility="collapsed",
            help=help,
            on_change=sync_from_input
        )

    return st.session_state[key]

def make_equity_chart(df, show_spy=True):
    chart_df = df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_df["date"],
        y=chart_df["total_equity"],
        mode="lines",
        name="AI Strategy",
        line=dict(width=3),
        hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>AI Strategy</b>: %{y:.2f}<extra></extra>"
    ))

    if show_spy and "spy_equity" in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=chart_df["date"],
            y=chart_df["spy_equity"],
            mode="lines",
            name="Market (SPY)",
            line=dict(width=2),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Market (SPY)</b>: %{y:.2f}<extra></extra>"
        ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        hovermode="x unified",
        dragmode="pan",
        xaxis=dict(
            type="date"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def apply_pending_config():
    if "pending_config" not in st.session_state:
        return

    cfg = st.session_state.pop("pending_config")

    st.session_state["selected_tickers"] = cfg.get(
        "tickers", st.session_state.get("selected_tickers", [])
    )
    st.session_state["asset_universe_mode"] = cfg.get(
        "asset_universe_mode",
        "Stocks + ETFs"
    )
    st.session_state["epochs"] = cfg.get(
        "epochs", st.session_state.get("epochs", DEFAULT_CONFIG["epochs"])
    )
    
    st.session_state["buy_top_n"] = cfg.get(
        "buy_top_n", st.session_state.get("buy_top_n", DEFAULT_CONFIG["buy_top_n"])
    )
    st.session_state["hold_top_n"] = cfg.get(
        "hold_top_n", st.session_state.get("hold_top_n", DEFAULT_CONFIG["hold_top_n"])
    )
    st.session_state["min_prob"] = cfg.get(
        "min_prob", st.session_state.get("min_prob", DEFAULT_CONFIG["min_prob"])
    )
    st.session_state["classification_threshold"] = cfg.get(
        "classification_threshold",
        st.session_state.get("classification_threshold", DEFAULT_CONFIG["classification_threshold"])
    )
    st.session_state["max_positions"] = cfg.get(
        "max_positions", st.session_state.get("max_positions", DEFAULT_CONFIG["max_positions"])
    )
    st.session_state["initial_capital"] = cfg.get(
        "initial_capital", st.session_state.get("initial_capital", DEFAULT_CONFIG["initial_capital"])
    )
    st.session_state["lookback_years"] = cfg.get(
        "lookback_years", st.session_state.get("lookback_years", DEFAULT_CONFIG["lookback_years"])
    )

    # fee slider shows percent, config stores decimal
    st.session_state["fee_per_trade"] = cfg.get(
        "fee_per_trade", DEFAULT_CONFIG["fee_per_trade"]
    ) * 100
    st.session_state["require_positive_return"] = cfg.get("require_positive_return", True)
    st.session_state["ranking_method"] = cfg.get("ranking_method", "Probability Only")
    
    st.session_state["require_positive_return"] = cfg.get("require_positive_return", True)

    # Convert backend value → UI label
    reverse_mapping = {
        "probability": "Probability Only",
        "return": "Predicted Return Only",
        "combined": "Combined Score"
    }
    st.session_state["ranking_method"] = reverse_mapping.get(
        cfg.get("ranking_method", "probability"),
        "Probability Only"
    )

    st.session_state["asset_universe_mode"] = cfg.get(
        "asset_universe_mode",
        "Stocks + ETFs"
    )
    

    if "min_expected_return" in cfg:
        st.session_state["min_expected_return_pct"] = cfg.get("min_expected_return", 0.0) * 100

    # if you use slider_with_input helper keys, sync them too
    for key in [
        "epochs", "buy_top_n", "hold_top_n", "min_prob",
        "classification_threshold", "max_positions",
        "initial_capital", "lookback_years",
        "fee_per_trade", "min_expected_return_pct"
    ]:
        slider_key = f"{key}_slider"
        input_key = f"{key}_input"

        if key in st.session_state:
            if slider_key in st.session_state:
                st.session_state[slider_key] = st.session_state[key]
            if input_key in st.session_state:
                st.session_state[input_key] = st.session_state[key]

st.set_page_config(page_title="ML Stock Dashboard", layout="wide")
st.title("AI-Driven Trading Experimentation Engine")

if "results" not in st.session_state:
    st.session_state["results"] = None

if "is_training" not in st.session_state:
    st.session_state["is_training"] = False

if "abort_training" not in st.session_state:
    st.session_state["abort_training"] = False

if "loaded_model_name" not in st.session_state:
    st.session_state["loaded_model_name"] = "None"

if "default_loaded" not in st.session_state:
    default_run = load_default_run()
    if default_run is not None:
        default_results = default_run.copy()
        default_config = default_results.pop("config", {})

        st.session_state["results"] = default_results
        st.session_state["active_config"] = default_config
        st.session_state["loaded_model_name"] = "default_model"
    st.session_state["default_loaded"] = True

if "active_config" not in st.session_state:
    st.session_state["active_config"] = DEFAULT_CONFIG.copy()

apply_pending_config()

loaded_name = st.session_state["loaded_model_name"]

st.markdown(
    f"""
    <div style="
        position: fixed;
        top: 20px;
        right: 30px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 8px 14px;
        border-radius: 12px;
        font-size: 0.85rem;
        color: #9ca3af;
        z-index: 9999;
    ">
        <span style="opacity:0.7;">Model:</span>
        <span style="color:white;font-weight:600;">{loaded_name}</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 18px;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        color: #9ca3af;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }

    .panel {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        color: white;
    }

    .subtle {
        color: #9ca3af;
        font-size: 0.92rem;
    }

    .green-text {
        color: #22c55e;
        font-weight: 700;
    }

    .red-text {
        color: #ef4444;
        font-weight: 700;
    }

    .amber-text {
        color: #f59e0b;
        font-weight: 700;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        height: 44px;
    }

    .stDataFrame, div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="panel">
    <div class="subtle">
        Train, compare, and paper trade machine learning stock strategies with saved models and live portfolio views.
    </div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown(
        '<div class="section-title" style="text-align:center;">Controls</div>',
        unsafe_allow_html=True
    )

    controls_locked = st.session_state["is_training"]

    simple_mode = st.toggle(
        "Simple Mode (recommended)",
        value=True,
        disabled=controls_locked
    )


    st.markdown("---")
    st.markdown(
        '<div class="section-title" style="text-align:center;">Stock Selection</div>',
        unsafe_allow_html=True
    )

    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = [
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

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Select All Stocks" if simple_mode else "Select All",
            disabled=controls_locked
        ):
            st.session_state.selected_tickers = ALL_TICKERS.copy()

    with col2:
        if st.button(
            "Clear Stock Selection" if simple_mode else "Deselect All",
            disabled=controls_locked
        ):
            st.session_state.selected_tickers = []

    selected_tickers = st.multiselect(
        "Choose stocks to train on" if simple_mode else "Choose tickers",
        options=ALL_TICKERS,
        default=st.session_state.selected_tickers,
        key="selected_tickers",
        disabled=controls_locked,
        help=(
            "Select which stocks the model will use for training and testing"
            if simple_mode
            else "Select ticker symbols used for model training and testing"
        )
    )

    st.caption(
        f"{len(selected_tickers)} stocks selected"
        if simple_mode
        else f"{len(selected_tickers)} tickers selected"
    )

    st.markdown("---")
    st.markdown(
        '<div class="section-title" style="text-align:center;">Model Configurations</div>',
        unsafe_allow_html=True
    )
    # st.markdown("### Model Configurations")

    ranking_method = st.selectbox(
        "Ranking Method",
        options=["Probability Only", "Predicted Return Only", "Combined Score"],
        index=0,
        key="ranking_method",
        disabled=controls_locked
    )

    asset_universe_mode = st.selectbox(
        "Asset Universe",
        options=[
            "Stocks Only",
            "Stocks + ETFs"
        ],
        index=1,
        key="asset_universe_mode",
        disabled=controls_locked,
        help="Choose whether the model can trade only stocks or stocks plus ETFs"
    )

    require_positive_return = st.toggle(
        "Require Positive Expected Return",
        value=True,
        key="require_positive_return",
        disabled=controls_locked,
        help="Only buy if predicted return is positive"
    )

    epochs = st.slider(
        "Training Length (How many learning iterations)" if simple_mode else "Epochs",
        5, 100, DEFAULT_CONFIG["epochs"],
        key="epochs",
        disabled=controls_locked,
        help="More training can improve learning, but may take longer"
    )

    
    lookback_years = st.slider(
        "Training Data Length (Years)" if simple_mode else "Lookback Years",
        1, 15, int(DEFAULT_CONFIG.get("lookback_years", 10)),
        key="lookback_years",
        disabled=controls_locked,
        help="How much historical data is used for training"
    )


    buy_top_n = st.slider(
        "How Many Stocks to Buy" if simple_mode else "Buy Top N",
        1, 20, DEFAULT_CONFIG["buy_top_n"],
        key="buy_top_n",
        disabled=controls_locked,
        help="Number of top-ranked stocks to buy"
    )

    hold_top_n = st.slider(
        "How Many Stocks to Keep" if simple_mode else "Hold Top N",
        1, 25, DEFAULT_CONFIG["hold_top_n"],
        key="hold_top_n",
        disabled=controls_locked,
        help="How many stocks can stay in the portfolio"
    )

    min_prob = slider_with_input(
        "Buy Confidence Threshold" if simple_mode else "Buy Threshold",
        0.0, 1.0, float(DEFAULT_CONFIG["min_prob"]), 0.01,
        key="min_prob",
        disabled=controls_locked,
        help="Only buy when the model is confident enough"
    )

    classification_threshold = slider_with_input(
        "Prediction Strictness" if simple_mode else "Classification Threshold",
        0.0, 1.0, float(DEFAULT_CONFIG["classification_threshold"]), 0.01,
        key="classification_threshold",
        disabled=controls_locked,
        help="Strictness of Up prediction. Higher values mean fewer, stricter positive predictions"
    )

    min_expected_return = slider_with_input(
    "Minimum Expected Return (%)" if simple_mode else "Min Expected Return",
    -2.0, 5.0, float(DEFAULT_CONFIG.get("min_expected_return", 0.0) * 100), 0.1,
    key="min_expected_return_pct",
    disabled=controls_locked,
    help="Only buy when the AI expects at least this return"
    ) / 100


    max_positions = st.slider(
        "Maximum Open Positions" if simple_mode else "Max Positions",
        1, 30, int(DEFAULT_CONFIG["max_positions"]),
        key="max_positions",
        disabled=controls_locked,
        help="Maximum number of stocks held at once"
    )

    initial_capital = st.number_input(
        "Starting Balance (£)" if simple_mode else "Initial Capital",
        min_value=100,
        max_value=1000000,
        value=int(DEFAULT_CONFIG["initial_capital"]),
        step=100,
        key="initial_capital",
        disabled=controls_locked,
        help="Starting portfolio cash"
    )

    fee_per_trade = slider_with_input(
        "Trading Fee (%)",
        0.0, 100.0, DEFAULT_CONFIG["fee_per_trade"] * 100, 0.1,
        key="fee_per_trade",
        disabled=controls_locked,
        help="Percentage cost applied to each trade"
    )

    fee_per_trade = fee_per_trade / 100

    st.markdown("---")
    st.markdown(
        '<div class="section-title" style="text-align:center;">Training Status</div>',
        unsafe_allow_html=True
    )

    run_button = st.button(
        "Start Training",
        disabled=controls_locked,
        use_container_width=True
    )

    # with btn2:
    #     abort_button = st.button(
    #         "Abort Training",
    #         disabled=not controls_locked,
    #         use_container_width=True
    #     )

    training_status_text = st.empty()
    training_progress_slot = st.empty()
    training_epoch_text = st.empty()


    st.markdown("---")
    st.markdown(
        '<div class="section-title" style="text-align:center;">Saved Models</div>',
        unsafe_allow_html=True
    )

    saved_runs = list_saved_runs()

    selected_saved_run = st.selectbox(
        "Load saved model",
        options=["None"] + saved_runs,
        disabled=controls_locked
    )

    selected_model_help = "Select a saved model to preview its parameters."

    if selected_saved_run != "None":
        loaded_preview = load_run(selected_saved_run)

        if loaded_preview is not None and "config" in loaded_preview:
            cfg = loaded_preview["config"]

            selected_model_help = (
                f"Epochs: {cfg.get('epochs', 'N/A')}\n"
                f"Buy Top N: {cfg.get('buy_top_n', 'N/A')}\n"
                f"Hold Top N: {cfg.get('hold_top_n', 'N/A')}\n"
                f"Min Prob: {cfg.get('min_prob', 'N/A')}\n"
                f"Classification Threshold: {cfg.get('classification_threshold', 'N/A')}\n"
                f"Max Positions: {cfg.get('max_positions', 'N/A')}\n"
                f"Initial Capital: {cfg.get('initial_capital', 'N/A')}\n"
                f"Fee Per Trade: {cfg.get('fee_per_trade', 'N/A')}\n"
                f"Lookback Years: {cfg.get('lookback_years', 'N/A')}\n"
                f"Min Expected Return: {cfg.get('min_expected_return', 'N/A')}\n"
                f"Stocks Used: {len(cfg.get('tickers', []))}"
            )
            
    st.caption("Hover over the button to preview parameters, then click to apply them.")

    apply_config_clicked = st.button(
        "Apply Configuration",
        disabled=(controls_locked or selected_saved_run == "None"),
        help=selected_model_help,
        use_container_width=True
    )

    if apply_config_clicked and selected_saved_run != "None":
        loaded_preview = load_run(selected_saved_run)

        if loaded_preview is not None and "config" in loaded_preview:
            st.session_state["pending_config"] = loaded_preview["config"]
            st.success(f"Applying configuration from '{selected_saved_run}'...")
            st.rerun()


    if selected_saved_run != "None":
        if st.button("Load Selected Model", disabled=controls_locked, use_container_width=True):
            loaded = load_run(selected_saved_run)
            if loaded is not None:
                loaded_results = loaded.copy()
                loaded_config = loaded_results.pop("config", {})

                st.session_state["results"] = loaded_results
                st.session_state["active_config"] = loaded_config
                st.session_state["loaded_model_name"] = selected_saved_run
                st.success(f"Loaded '{selected_saved_run}'")
    model_name = st.text_input("Save Model As").strip() or "latest_run"
    if st.button("Save Current Model", disabled=controls_locked, use_container_width=True):
        

        if st.session_state["results"] is None:
            st.error("No run available to save.")
        elif not model_name:
            st.error("Enter a save name first.")
        elif model_name in saved_runs:
            st.error(f"A model named '{model_name}' already exists. Choose a different name.")
        else:
            current_config = DEFAULT_CONFIG.copy()
            current_config["tickers"] = st.session_state["selected_tickers"]
            current_config["epochs"] = st.session_state["epochs"]
            current_config["buy_top_n"] = st.session_state["buy_top_n"]
            current_config["hold_top_n"] = st.session_state["hold_top_n"]
            current_config["min_prob"] = st.session_state["min_prob"]
            current_config["classification_threshold"] = st.session_state["classification_threshold"]
            current_config["max_positions"] = st.session_state["max_positions"]
            current_config["fee_per_trade"] = st.session_state["fee_per_trade"] / 100
            current_config["initial_capital"] = st.session_state["initial_capital"]
            current_config["lookback_years"] = st.session_state["lookback_years"]

            save_run(model_name, st.session_state["results"], current_config)
            st.success(f"Saved as '{model_name}'")

if st.session_state.get("pending_training_start", False):
    st.session_state["pending_training_start"] = False

    config = DEFAULT_CONFIG.copy()
    config["epochs"] = st.session_state["epochs"]
    config["buy_top_n"] = st.session_state["buy_top_n"]
    config["hold_top_n"] = st.session_state["hold_top_n"]
    config["min_prob"] = st.session_state["min_prob"]
    config["classification_threshold"] = st.session_state["classification_threshold"]
    config["min_expected_return"] = st.session_state["min_expected_return_pct"] / 100
    config["max_positions"] = st.session_state["max_positions"]
    config["fee_per_trade"] = st.session_state["fee_per_trade"] / 100
    config["initial_capital"] = st.session_state["initial_capital"]
    config["lookback_years"] = st.session_state["lookback_years"]
    config["tickers"] = st.session_state["selected_tickers"]
    config["require_positive_return"] = st.session_state["require_positive_return"]

    mapping = {
        "Probability Only": "probability",
        "Predicted Return Only": "return",
        "Combined Score": "combined"
    }
    config["ranking_method"] = mapping[st.session_state["ranking_method"]]
    config["asset_universe_mode"] = st.session_state["asset_universe_mode"]

    progress_bar = training_progress_slot.progress(0)
    training_status_text.info("Training in progress...")
    training_epoch_text.caption("Preparing model...")

    def update_progress(current_epoch, total_epochs):
        percent = int((current_epoch / total_epochs) * 100)
        progress_bar.progress(percent)
        training_epoch_text.caption(f"Epoch {current_epoch} / {total_epochs} ({percent}%)")

    try:
        with st.spinner("Training model..."):
            st.session_state["results"] = run_experiment(
                config,
                progress_callback=update_progress
            )

        st.session_state["active_config"] = config.copy()

        st.session_state["loaded_model_name"] = "latest_run"
        training_status_text.success("Training complete.")
        training_epoch_text.caption(f"Finished {config['epochs']} epochs.")
        progress_bar.progress(100)

    except Exception as e:
        training_status_text.error("Training failed.")
        training_epoch_text.caption("An error occurred during training.")
        progress_bar.progress(0)
        st.error(f"Error during training: {e}")

    finally:
        st.session_state["is_training"] = False

        if st.session_state["results"] is not None:
            save_run("latest_run", st.session_state["results"], config)

        st.rerun()


# if abort_button and st.session_state["is_training"]:
#     st.session_state["abort_training"] = True


if epochs < 15:
    st.warning("Very short training may lead to poor learning")

if epochs > 80:
    st.warning("Long training may cause overfitting")

if epochs > 120:
    st.error("Extremely long training is likely overfitting and wasting time")

if fee_per_trade > 0.01:
    st.warning("High trading fees may significantly reduce profits")

if fee_per_trade > 0.02:
    st.error("Very high fees may make the strategy unprofitable")

if min_prob < 0.5:
    st.warning("Low buy confidence may lead to weak trades")

if min_prob > 0.8:
    st.info("Very selective strategy may produce fewer but stronger trades")

if classification_threshold > min_prob:
    st.warning("Prediction threshold is stricter than buy threshold, which may create conflicting behaviour")

if max_positions < 3:
    st.warning("A very concentrated portfolio increases risk")

if max_positions > 20:
    st.info("A highly diversified portfolio spreads risk but reduces impact per trade")

if buy_top_n > max_positions:
    st.error("Buy count is higher than the maximum allowed positions")

if buy_top_n >= 15 and min_prob < 0.6:
    st.warning("This is an aggressive strategy with many lower-confidence trades")

if buy_top_n <= 3 and min_prob > 0.75:
    st.info("This is a highly selective strategy with few high-confidence trades")

if lookback_years < 3:
    st.warning("Very little historical data may reduce model reliability")

if lookback_years > 12:
    st.info("A longer training history may improve robustness but increase runtime")

if fee_per_trade > 0.01 and buy_top_n > 10:
    st.warning("High fees combined with frequent trading may reduce profits significantly")

if min_prob < 0.55 and classification_threshold < 0.55:
    st.warning("Loose confidence settings may produce too many false positives")

if max_positions < 5 and buy_top_n > 10:
    st.warning("Trying to buy many stocks while holding very few may make the strategy inefficient")
    


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
    if len(selected_tickers) == 0:
        st.error("Please select at least one ticker.")
    else:
        st.session_state["is_training"] = True
        st.session_state["pending_training_start"] = True
        st.session_state["results"] = None
        st.rerun()

        config = DEFAULT_CONFIG.copy()
        config["epochs"] = st.session_state["epochs"]
        config["buy_top_n"] = st.session_state["buy_top_n"]
        config["hold_top_n"] = st.session_state["hold_top_n"]
        config["min_prob"] = st.session_state["min_prob"]
        config["classification_threshold"] = st.session_state["classification_threshold"]
        config["min_expected_return"] = st.session_state["min_expected_return_pct"] / 100
        config["max_positions"] = st.session_state["max_positions"]
        config["fee_per_trade"] = st.session_state["fee_per_trade"] / 100
        config["initial_capital"] = st.session_state["initial_capital"]
        config["lookback_years"] = st.session_state["lookback_years"]
        config["tickers"] = st.session_state["selected_tickers"]
        config["require_positive_return"] = st.session_state["require_positive_return"]
        config["asset_universe_mode"] = st.session_state["asset_universe_mode"]

        mapping = {
            "Probability Only": "probability",
            "Predicted Return Only": "return",
            "Combined Score": "combined"
        }
        config["ranking_method"] = mapping[st.session_state["ranking_method"]]

        config["asset_universe_mode"] = st.session_state["asset_universe_mode"]

        progress_bar = training_progress_slot.progress(0)
        training_status_text.info("Training in progress...")
        training_epoch_text.caption("Preparing model...")

        def update_progress(current_epoch, total_epochs):
            if st.session_state.get("abort_training", False):
                raise KeyboardInterrupt("Training aborted by user.")

            percent = int((current_epoch / total_epochs) * 100)
            progress_bar.progress(percent)
            training_epoch_text.caption(f"Epoch {current_epoch} / {total_epochs} ({percent}%)")

        try:
            with st.spinner("Training model..."):
                st.session_state["results"] = run_experiment(
                    config,
                    progress_callback=update_progress
                )
            
            st.session_state["loaded_model_name"] = "latest_run"

            training_status_text.success("Training complete.")
            training_epoch_text.caption(f"Finished {epochs} epochs.")
            progress_bar.progress(100)

        except KeyboardInterrupt:
            training_status_text.warning("Training aborted.")
            training_epoch_text.caption("Training stopped before completion.")
            training_progress_slot.empty()
            st.warning("Training was aborted before completion.")


        except Exception as e:
            training_status_text.error("Training failed.")
            training_epoch_text.caption("An error occurred during training.")
            training_progress_slot.empty()
            st.error(f"Error during training: {e}")

        finally:
            st.session_state["is_training"] = False
            st.session_state["abort_training"] = False

            if st.session_state["results"] is not None:
                auto_name = "latest_run"
                save_run(auto_name, st.session_state["results"], config)

results = st.session_state["results"]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard",
    "Paper Trading",
    "Portfolio",
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
    holdings_df = results.get("holdings_df", pd.DataFrame())

    with tab1:
        st.subheader("Dashboard")

        st.caption(f"Showing configuration for: {st.session_state.get('loaded_model_name', 'current run')}")

        active_cfg = st.session_state.get("active_config", DEFAULT_CONFIG)

        ranking_map = {
            "probability": "Probability Only",
            "return": "Predicted Return Only",
            "combined": "Combined Score"
        }

        ranking = ranking_map.get(active_cfg.get("ranking_method", "probability"), "N/A")
        min_prob = active_cfg.get("min_prob", 0.0)
        classification_threshold = active_cfg.get("classification_threshold", 0.0)
        min_return = active_cfg.get("min_expected_return", 0.0) * 100
        universe = active_cfg.get("asset_universe_mode", "N/A")
        max_pos = active_cfg.get("max_positions", 0)
        require_pos = active_cfg.get("require_positive_return", False)

        buy_top_n = active_cfg.get("buy_top_n", 0)
        hold_top_n = active_cfg.get("hold_top_n", 0)
        initial_capital = active_cfg.get("initial_capital", 0)
        fee_pct = active_cfg.get("fee_per_trade", 0.0) * 100
        epochs = active_cfg.get("epochs", 0)
        lookback_years = active_cfg.get("lookback_years", 0)
        num_tickers = len(active_cfg.get("tickers", []))

        st.markdown(f"""
        
        <div style="
            padding: 12px;
            border-radius: 10px;
            background-color: #0f172a;
            border: 1px solid #334155;
            font-size: 0.9rem;
        ">

        <div style="
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        ">

        <div><b>Ranking:</b> {ranking}</div>
        <div><b>Buy Confidence:</b> {min_prob:.2f}</div>
        <div><b>Prediction Strictness:</b> {classification_threshold:.2f}</div>

        <div><b>Min Return:</b> {min_return:.2f}%</div>
        <div><b>Positive Only:</b> {"Yes" if require_pos else "No"}</div>
        <div><b>Universe:</b> {universe}</div>

        <div><b>Tickers:</b> {num_tickers}</div>
        <div><b>Buy N:</b> {buy_top_n}</div>
        <div><b>Hold N:</b> {hold_top_n}</div>

        <div><b>Max Pos:</b> {max_pos}</div>
        <div><b>Capital:</b> £{initial_capital:,}</div>
        <div><b>Fee:</b> {fee_pct:.2f}%</div>

        <div><b>Epochs:</b> {epochs}</div>
        <div><b>Lookback:</b> {lookback_years}y</div>

        </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f'{metrics["test_accuracy"]:.3f}')
        c2.metric("Portfolio Value", f'${metrics["final_portfolio_value"]:.2f}')
        c3.metric("Total Return", f'{metrics["total_return"] * 100:.2f}%')
        c4.metric("Max Drawdown", f'{metrics["max_drawdown"] * 100:.2f}%')


        header_with_help("Equity Curve", "Shows how the AI strategy portfolio value changed over time. You can compare it against SPY to see whether the strategy outperformed the market.")

        
        show_spy_db = st.checkbox(
            "Compare with Market (SPY)",
            value=True,
            key="show_spy_db"
        )
        
        equity_df = portfolio_history_df[["date", "total_equity"]].copy()

        if show_spy_db:
            equity_df = equity_df.merge(
                spy_benchmark_df[["date", "spy_equity"]],
                on="date",
                how="left"
            )

        st.plotly_chart(
            make_equity_chart(equity_df, show_spy=show_spy_db),
            use_container_width=True, config = PLOTLY_CONFIG,
            key="dashboard_equity_chart"
        )
        # equity_df = equity_df.set_index("date")

        # st.line_chart(equity_df)

        header_with_help(
            "Top Picks",
            "Displays the highest-ranked assets based on your selected ranking method and filters."
        )

        top_picks_chart_df = top_picks_df.copy()

        top_picks_chart_df["Buy Score"] = (top_picks_chart_df["prob_up"] * 100).round(1)
        top_picks_chart_df["Predicted Return %"] = (top_picks_chart_df["predicted_return"] * 100).round(2)
        top_picks_chart_df["Predicted Price"] = top_picks_chart_df["predicted_price"].round(2)
        top_picks_chart_df["Confidence"] = ((top_picks_chart_df["prob_up"] - 0.5) * 200).round(1)
        top_picks_chart_df["Risk"] = (top_picks_chart_df.get("volatility_20", pd.Series(0, index=top_picks_chart_df.index)) * 100).round(2)
        top_picks_chart_df["label"] = top_picks_chart_df["Ticker"]
        top_picks_chart_df["Price"] = top_picks_chart_df["Close"].round(2)

        if "rank" not in top_picks_chart_df.columns:
            top_picks_chart_df["rank"] = range(1, len(top_picks_chart_df) + 1)

        top_picks_chart_df["Rank"] = top_picks_chart_df["rank"]


        top_pick_metric = st.selectbox(
            "Top Picks Chart Metric",
            options=["Buy Score", "Predicted Return %", "Predicted Price", "Confidence", "Price", "Rank"],
            index=0,
            key="top_pick_metric"
        )

        if top_pick_metric == "Rank":
            st.caption("Lower rank is better (Rank 1 = strongest pick).")

        metric_titles = {
            "Buy Score": "AI Buy Score",
            "Predicted Return %": "Predicted Return (%)",
            "Predicted Price": "Predicted Price ($)",
            "Confidence": "Confidence",
            "Price": "Current Price ($)",
            "Rank": "Rank"
        }

        chart = (
            alt.Chart(top_picks_chart_df)
            .mark_bar(cornerRadiusEnd=8)
            .encode(
                y=alt.Y(
                    "label:N",
                    sort=("-x" if top_pick_metric != "Rank" else "x"),
                    title=None
                ),
                x=alt.X(f"{top_pick_metric}:Q", title=metric_titles[top_pick_metric]),
                color=alt.Color(
                    "label:N",
                    legend=None,
                    scale=alt.Scale(scheme="tableau10")
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Stock"),
                    alt.Tooltip("Buy Score:Q", title="Buy Score", format=".1f"),
                    alt.Tooltip("Predicted Return %:Q", title="Predicted Return %", format=".2f"),
                    alt.Tooltip("Predicted Price:Q", title="Predicted Price", format=".2f"),
                    alt.Tooltip("Confidence:Q", title="Confidence", format=".1f"),
                    alt.Tooltip("Price:Q", title="Current Price", format=".2f"),
                    alt.Tooltip("Rank:Q", title="Rank", format=".0f")
                ]
            )
            .properties(height=320)
        )

        st.altair_chart(chart, use_container_width=True)

        header_with_help("Strongest Signals", "Quick snapshot of the strongest current buy signals based on the model's predictions.")

        summary_cols = st.columns(min(len(top_picks_chart_df), 5))
        for i, (_, row) in enumerate(top_picks_chart_df.head(5).iterrows()):
            summary_cols[i].metric(
                row["Ticker"],
                f"{row['Buy Score']:.1f}%"
            )

    with tab2: 
        st.subheader("Paper Trading")

        header_with_help(
            "Paper Trading Overview",
            "Summarizes the simulated portfolio performance using the AI strategy over time."
        )

        c1, c2, c3 = st.columns(3)

        initial_capital_val = st.session_state.get("active_config", DEFAULT_CONFIG).get("initial_capital", 1000)
        final_portfolio_value = metrics["final_portfolio_value"]
        total_return_pct = metrics["total_return"] * 100

        portfolio_value_class = get_change_class(final_portfolio_value, initial_capital_val)

        if total_return_pct > 0:
            total_return_class = "green-text"
        elif total_return_pct < 0:
            total_return_class = "red-text"
        else:
            total_return_class = "amber-text"

        with c1:
            metric_card("Paper Portfolio Value", f"${final_portfolio_value:.2f}", portfolio_value_class)

        with c2:
            metric_card("Open Positions", f'{metrics["open_positions_end"]}')

        with c3:
            metric_card("Total Return", f"{total_return_pct:.2f}%", total_return_class)

        header_with_help(
            "Paper Equity Curve",
            "Shows how your models (paper trading) portfolio value changes over time based on the AI's buy and sell decisions."
        )

        paper_df = portfolio_history_df[["date", "total_equity"]].copy()
        paper_df["date"] = pd.to_datetime(paper_df["date"])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=paper_df["date"],
            y=paper_df["total_equity"],
            mode="lines",
            name="Paper Portfolio",
            line=dict(width=3),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Portfolio Value</b>: %{y:.2f}<extra></extra>"
        ))

        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode="x unified",
            dragmode="pan",
            xaxis=dict(type="date")
        )

        st.plotly_chart(fig, use_container_width=True, config = PLOTLY_CONFIG, key="paper_equity_chart")

        header_with_help(
            "AI Buy Signals",
            "Shows the assets the model would currently buy based on predictions and filters."
        )

        buy_signals_df = top_picks_df.copy()
        buy_signals_df["Stock"] = buy_signals_df["Ticker"]
        buy_signals_df["Price"] = buy_signals_df["Close"].round(2)

        if "rank" not in buy_signals_df.columns:
            buy_signals_df["rank"] = range(1, len(buy_signals_df) + 1)
    
        buy_signals_df["Rank"] = buy_signals_df["rank"]

        buy_signals_df["Predicted Return %"] = (buy_signals_df["predicted_return"] * 100).round(2)
        buy_signals_df["Predicted Price Tomorrow"] = buy_signals_df["predicted_price"].round(2)

        buy_signals_display = buy_signals_df[
            ["Stock", "Price", "Rank", "Predicted Return %", "Predicted Price Tomorrow"]
        ].copy()

        styled_buy_signals = (
            buy_signals_display.style
            .map(highlight_positive_negative, subset=["Predicted Return %"])
            .format({
                "Price": "{:.2f}",
                "Predicted Return %": "{:.2f}%",
                "Predicted Price Tomorrow": "{:.2f}"
            })
        )

        # Compare predicted price tomorrow vs current price
        predicted_price_styles = []
        for _, row in buy_signals_display.iterrows():
            if row["Predicted Price Tomorrow"] > row["Price"]:
                predicted_price_styles.append("color: #22c55e; font-weight: 600;")
            elif row["Predicted Price Tomorrow"] < row["Price"]:
                predicted_price_styles.append("color: #ef4444; font-weight: 600;")
            else:
                predicted_price_styles.append("color: #f59e0b; font-weight: 600;")

        styled_buy_signals = styled_buy_signals.apply(
            lambda _: predicted_price_styles,
            subset=["Predicted Price Tomorrow"]
        )

        st.dataframe(styled_buy_signals, use_container_width=True, hide_index=True)

        with st.expander("More"):
            st.dataframe(buy_signals_df, use_container_width=True)

        header_with_help(
            "Recent Paper Trades",
            "Lists recent simulated buy and sell actions taken by the model."
        )

        paper_trades_df = trade_log_df.copy()

        if "date" in paper_trades_df.columns:
            paper_trades_df["date"] = pd.to_datetime(paper_trades_df["date"]).dt.strftime("%Y-%m-%d")

        if "profit" not in paper_trades_df.columns:
            paper_trades_df["profit"] = None

        if "profit_pct" not in paper_trades_df.columns:
            paper_trades_df["profit_pct"] = None

        paper_trades_display = paper_trades_df[["date", "ticker", "action", "rank", "value", "profit", "profit_pct"]].copy()

        paper_trades_display["value"] = pd.to_numeric(paper_trades_display["value"], errors="coerce").round(2)
        paper_trades_display["profit"] = pd.to_numeric(paper_trades_display["profit"], errors="coerce").round(2)
        paper_trades_display["profit_pct"] = pd.to_numeric(paper_trades_display["profit_pct"], errors="coerce").round(2)

        paper_trades_display.columns = ["Date", "Stock", "Action", "Rank", "Value", "P/L", "P/L %"]

        def highlight_action(val):
            if val == "BUY":
                return "color: #22c55e; font-weight: 600;"
            elif val == "SELL":
                return "color: #ef4444; font-weight: 600;"
            return ""
        
        def highlight_profit(val):
            if pd.isna(val):
                return ""
            elif val > 0:
                return "color: #22c55e; font-weight: 600;"
            elif val < 0:
                return "color: #ef4444; font-weight: 600;"
            return ""

        styled_trades = (
            paper_trades_display.tail(25)
            .style
            .map(highlight_action, subset=["Action"])
            .map(highlight_profit, subset=["P/L"])
            .map(highlight_profit, subset=["P/L %"])
            .format({
                "Value": "{:.2f}",
                "P/L": "{:.2f}",
                "P/L %": "{:.2f}%"
            })
        )

        st.dataframe(styled_trades, use_container_width=True, hide_index=True)

        with st.expander("More details"):
            st.dataframe(paper_trades_df.head(100), use_container_width=True)

    with tab3:
        st.subheader("Portfolio")

        if not portfolio_history_df.empty:
            start_date = pd.to_datetime(portfolio_history_df["date"]).min()
            end_date = pd.to_datetime(portfolio_history_df["date"]).max()
            total_days = (end_date - start_date).days
            total_years = total_days / 365.25

            start_value = float(portfolio_history_df["total_equity"].iloc[0])
            end_value = float(portfolio_history_df["total_equity"].iloc[-1])
            pnl = end_value - start_value
            pnl_pct = ((end_value / start_value) - 1) * 100 if start_value != 0 else 0

            c1, c2, c3 = st.columns(3)

            net_change_class = get_change_class(end_value, start_value)

            with c1:
                metric_card("Backtest Period", f"{total_years:.1f} years")

            with c2:
                metric_card("Start Value", f"${start_value:.2f}")

            with c3:
                metric_card("Net Change", f"${pnl:.2f} ({pnl_pct:.2f}%)", net_change_class)


        if holdings_df is not None and not holdings_df.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Holdings Count", len(holdings_df))
            c2.metric("Largest Position", f'${holdings_df["position_value"].max():.2f}')
            c3.metric("Total Held Value", f'${holdings_df["position_value"].sum():.2f}')
            
            header_with_help(
                "Open Positions",
                "Shows all currently held assets and their values in the simulated portfolio."
            )
            
            cols = list(holdings_df.columns)

            if "latest_rank" in cols:
                cols = ["latest_rank"] + [c for c in cols if c != "latest_rank"]

            holdings_df = holdings_df[cols]

            display_holdings_df = holdings_df.copy()

            if "prob_up" in display_holdings_df.columns:
                display_holdings_df = display_holdings_df.drop(columns=["prob_up"])

            if "predicted_return" in display_holdings_df.columns:
                display_holdings_df["Predicted Return %"] = (display_holdings_df["predicted_return"] * 100).round(2)
                display_holdings_df = display_holdings_df.drop(columns=["predicted_return"])

            if "predicted_price" in display_holdings_df.columns:
                display_holdings_df["Predicted Price"] = display_holdings_df["predicted_price"].round(2)
                display_holdings_df = display_holdings_df.drop(columns=["predicted_price"])

            if "position_value" in display_holdings_df.columns:
                display_holdings_df["position_value"] = display_holdings_df["position_value"].round(2)

            if "entry_price" in display_holdings_df.columns:
                display_holdings_df["entry_price"] = display_holdings_df["entry_price"].round(2)

            if "current_price" in display_holdings_df.columns:
                display_holdings_df["current_price"] = display_holdings_df["current_price"].round(2)

            rename_map = {
                "latest_rank": "Rank",
                "ticker": "Ticker",
                "shares": "Shares",
                "entry_price": "Entry Price",
                "current_price": "Current Price",
                "position_value": "Position Value"
            }
            display_holdings_df = display_holdings_df.rename(columns=rename_map)

            preferred_order = [
                "Rank",
                "Ticker",
                "Shares",
                "Entry Price",
                "Current Price",
                "Predicted Return %",
                "Predicted Price",
                "Position Value"
            ]

            final_cols = [col for col in preferred_order if col in display_holdings_df.columns] + [
                col for col in display_holdings_df.columns if col not in preferred_order
            ]

            display_holdings_df = display_holdings_df[final_cols]

            st.dataframe(display_holdings_df, use_container_width=True, hide_index=True)

        else:
            st.info("No open holdings in the current AI portfolio.")
        return_pct = metrics["total_return"] * 100
        return_class = "green-text" if return_pct >= 0 else "red-text"


        st.markdown(
            f'<div class="panel"><div class="section-title">Performance</div>'
            f'<div class="{return_class}" style="font-size:2rem;">{return_pct:.2f}%</div></div>',
            unsafe_allow_html=True
)


    with tab4:
        st.subheader("Live Signals")

        header_with_help(
            "Buy Now",
            "Assets the model would currently buy based on latest predictions."
        )

        buy_now_df = top_picks_df.copy()
        

        if "rank" not in buy_now_df.columns:
            buy_now_df["rank"] = range(1, len(buy_now_df) + 1)

        buy_now_display = buy_now_df[
            ["rank", "Ticker", "Close", "prob_up", "predicted_return", "predicted_price"]
        ].copy()

        buy_now_display = buy_now_display.rename(columns={
            "rank": "Rank",
            "Ticker": "Ticker",
            "Close": "Current Price",
            "prob_up": "Buy Score",
            "predicted_return": "Predicted Return %",
            "predicted_price": "Predicted Price"
        })

        buy_now_display["Current Price"] = buy_now_display["Current Price"].round(2)
        buy_now_display["Buy Score"] = (buy_now_display["Buy Score"] * 100).round(2)
        buy_now_display["Predicted Return %"] = (buy_now_display["Predicted Return %"] * 100).round(2)
        buy_now_display["Predicted Price"] = buy_now_display["Predicted Price"].round(2)

        styled_buy_now = (
            buy_now_display.style
            .map(highlight_positive_negative, subset=["Predicted Return %"])
            .format({
                "Current Price": "{:.2f}",
                "Buy Score": "{:.2f}%",
                "Predicted Return %": "{:.2f}%",
                "Predicted Price": "{:.2f}"
            })
        )

        buy_now_price_styles = []
        for _, row in buy_now_display.iterrows():
            if row["Predicted Price"] > row["Current Price"]:
                buy_now_price_styles.append("color: #22c55e; font-weight: 600;")
            elif row["Predicted Price"] < row["Current Price"]:
                buy_now_price_styles.append("color: #ef4444; font-weight: 600;")
            else:
                buy_now_price_styles.append("color: #f59e0b; font-weight: 600;")

        styled_buy_now = styled_buy_now.apply(
            lambda _: buy_now_price_styles,
            subset=["Predicted Price"]
        )

        st.dataframe(styled_buy_now, use_container_width=True, hide_index=True)

        st.markdown("### Current Portfolio Snapshot")
        c1, c2, c3 = st.columns(3)
        c1.metric("Open Positions", metrics["open_positions_end"])
        c2.metric("Buy Trades", metrics["num_buy_trades"])
        c3.metric("Sell Trades", metrics["num_sell_trades"])

        recent_actions_df = trade_log_df.copy()

        if "date" in recent_actions_df.columns:
            recent_actions_df["date"] = pd.to_datetime(recent_actions_df["date"]).dt.strftime("%Y-%m-%d")

        if "profit" not in recent_actions_df.columns:
            recent_actions_df["profit"] = None

        if "profit_pct" not in recent_actions_df.columns:
            recent_actions_df["profit_pct"] = None

        recent_actions_display = recent_actions_df[
            ["date", "action", "ticker", "value", "profit", "profit_pct"]
        ].copy()

        recent_actions_display["value"] = pd.to_numeric(
            recent_actions_display["value"], errors="coerce"
        ).round(2)

        recent_actions_display["profit"] = pd.to_numeric(
            recent_actions_display["profit"], errors="coerce"
        ).round(2)

        recent_actions_display["profit_pct"] = pd.to_numeric(
            recent_actions_display["profit_pct"], errors="coerce"
        ).round(2)

        recent_actions_display = recent_actions_display.rename(columns={
            "date": "Date",
            "action": "Action",
            "ticker": "Ticker",
            "value": "Value",
            "profit": "P/L",
            "profit_pct": "P/L %"
        })

        styled_actions = (
            recent_actions_display.head(10)
            .style
            .map(highlight_action, subset=["Action"])
            .map(highlight_profit, subset=["P/L"])
            .map(highlight_profit, subset=["P/L %"])
            .format({
                "Value": "{:.2f}",
                "P/L": "{:.2f}",
                "P/L %": "{:.2f}%"
            })
        )

        st.dataframe(styled_actions, use_container_width=True, hide_index=True)

    with tab5:
        st.subheader("Backtesting")

        show_spy_backtesting = st.checkbox(
            "Compare with Market (SPY)",
            value=True,
            key="show_spy_backtesting"
        )

        equity_df = portfolio_history_df[["date", "total_equity"]].copy()

        if show_spy_backtesting:
            equity_df = equity_df.merge(
                spy_benchmark_df[["date", "spy_equity"]],
                on="date",
                how="left"
            )

        st.plotly_chart(
            make_equity_chart(equity_df, show_spy=show_spy_backtesting),
            use_container_width=True, config = PLOTLY_CONFIG, key="backtest_equity_chart"
        )

        st.markdown("### Trade Log")

        backtest_trades_df = trade_log_df.copy()

        if "date" in backtest_trades_df.columns:
            backtest_trades_df["date"] = pd.to_datetime(backtest_trades_df["date"]).dt.strftime("%Y-%m-%d")

        if "profit" not in backtest_trades_df.columns:
            backtest_trades_df["profit"] = None

        if "profit_pct" not in backtest_trades_df.columns:
            backtest_trades_df["profit_pct"] = None

        for col in ["value", "profit", "profit_pct"]:
            if col in backtest_trades_df.columns:
                backtest_trades_df[col] = pd.to_numeric(backtest_trades_df[col], errors="coerce")

        styled_backtest_trades = (
            backtest_trades_df.style
            .map(highlight_action, subset=["action"] if "action" in backtest_trades_df.columns else None)
            .map(highlight_profit, subset=["profit"] if "profit" in backtest_trades_df.columns else None)
            .map(highlight_profit, subset=["profit_pct"] if "profit_pct" in backtest_trades_df.columns else None)
            .format({
                "value": "{:.2f}",
                "profit": "{:.2f}",
                "profit_pct": "{:.2f}%"
            })
        )

        st.dataframe(styled_backtest_trades, use_container_width=True, hide_index=True)

        st.markdown("### Portfolio History")

        portfolio_history_display = portfolio_history_df.head(30).copy()

        # remove drawdown column if it exists
        if "drawdown" in portfolio_history_display.columns:
            portfolio_history_display = portfolio_history_display.drop(columns=["drawdown"])

        numeric_cols = ["total_equity", "cash", "holdings_value", "peak_equity"]

        for col in numeric_cols:
            if col in portfolio_history_display.columns:
                portfolio_history_display[col] = pd.to_numeric(portfolio_history_display[col], errors="coerce")

        styled_portfolio_history = (
            portfolio_history_display.style
            .apply(highlight_price_change_from_prev, subset=["total_equity"])
            .apply(highlight_price_change_from_prev, subset=["cash"])
            .apply(highlight_price_change_from_prev, subset=["holdings_value"])
            .apply(highlight_price_change_from_prev, subset=["peak_equity"])
            .format({
                "total_equity": "{:.2f}",
                "cash": "{:.2f}",
                "holdings_value": "{:.2f}",
                "peak_equity": "{:.2f}",
            })
        )

        st.dataframe(styled_portfolio_history, use_container_width=True, hide_index=True)

    with tab6:
        st.subheader("Training")

        final_epoch = int(training_history_df["epoch"].iloc[-1])
        final_loss = float(training_history_df["loss"].iloc[-1])
        final_train_acc = float(training_history_df["train_accuracy"].iloc[-1])

        c1, c2, c3 = st.columns(3)
        c1.metric("Epochs Completed", final_epoch)
        c2.metric("Final Loss", f"{final_loss:.4f}")
        c3.metric("Final Train Accuracy", f"{final_train_acc:.3f}")

        st.markdown("### Training Loss")

        loss_df = training_history_df.copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=loss_df["epoch"],
            y=loss_df["loss"],
            mode="lines+markers",
            name="Loss",
            line=dict(width=3),
            hovertemplate="<b>Epoch</b>: %{x}<br><b>Loss</b>: %{y:.4f}<extra></extra>"
        ))

        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True, config = PLOTLY_CONFIG)

        st.markdown("### Training Accuracy")

        acc_df = training_history_df.copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=acc_df["epoch"],
            y=acc_df["train_accuracy"],
            mode="lines+markers",
            name="Train Accuracy",
            line=dict(width=3),
            hovertemplate="<b>Epoch</b>: %{x}<br><b>Train Accuracy</b>: %{y:.3f}<extra></extra>"
        ))

        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Epoch",
            yaxis_title="Train Accuracy",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True, config = PLOTLY_CONFIG)

        st.markdown("### Feature Importance")
        st.dataframe(feature_importance_df.head(20), use_container_width=True, hide_index= True)

        st.markdown("### Prediction Samples")
        st.dataframe(predictions_df.tail(30), use_container_width=True)
        



else:
    st.info("Adjust the settings and click Start Training.")