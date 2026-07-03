import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import warnings
from scipy.optimize import minimize
from scipy import stats

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURAÇÃO DA PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Equity Allocator Pro",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# ── Paleta Corporativa ────────────────────────────────────────────────────────
CORP = {
    "primary":    "#1A56DB",
    "success":    "#0E9F6E",
    "warning":    "#E3A008",
    "danger":     "#E02424",
    "purple":     "#7E3AF2",
    "neutral":    "#6B7280",
    "bg_card":    "#FFFFFF",
    "bg_page":    "#F8FAFC",
    "border":     "#E2E8F0",
    "text_main":  "#0F172A",
    "text_sub":   "#64748B",
}

# ── Paleta de cores para ações individuais (até 20) ───────────────────────────
STOCK_COLORS = [
    "#1A56DB", "#0E9F6E", "#E3A008", "#E02424", "#7E3AF2",
    "#0694A2", "#C27803", "#3F83F8", "#D61F69", "#057A55",
    "#6C2BD9", "#B45309", "#2563EB", "#BE185D", "#059669",
    "#7C3AED", "#D97706", "#1D4ED8", "#9D174D", "#047857",
]

# ── CSS Premium ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stApp {{ background-color: {CORP['bg_page']}; }}
  .metric-card {{
    background: {CORP['bg_card']};
    border: 1px solid {CORP['border']};
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    text-align: center;
    transition: box-shadow 0.2s ease;
  }}
  .metric-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.10); }}
  .metric-value {{
    font-size: 26px; font-weight: 700; color: {CORP['text_main']};
    letter-spacing: -0.5px; line-height: 1.2;
  }}
  .metric-value.positive {{ color: {CORP['success']}; }}
  .metric-value.negative {{ color: {CORP['danger']}; }}
  .metric-label {{
    font-size: 11px; font-weight: 600; color: {CORP['text_sub']};
    text-transform: uppercase; letter-spacing: 0.6px; margin-top: 4px;
  }}
  .metric-accent {{ height: 3px; border-radius: 2px; margin: 10px auto 0; width: 40px; }}
  .section-header {{
    border-left: 4px solid {CORP['primary']};
    padding-left: 10px; margin: 24px 0 12px;
    font-size: 16px; font-weight: 700; color: {CORP['text_main']};
  }}
  .badge {{
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.3px;
  }}
  .badge-ok   {{ background:#D1FAE5; color:#065F46; }}
  .badge-err  {{ background:#FEE2E2; color:#991B1B; }}
  .badge-warn {{ background:#FEF3C7; color:#92400E; }}
  .stTabs [data-baseweb="tab-list"] {{
    gap: 4px; background: {CORP['bg_card']};
    border-radius: 10px 10px 0 0; padding: 6px 6px 0;
    border-bottom: 2px solid {CORP['border']};
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 8px 8px 0 0; padding: 8px 18px;
    font-size: 13px; font-weight: 600;
    color: {CORP['text_sub']}; background: transparent; border: none; transition: color 0.15s;
  }}
  .stTabs [aria-selected="true"] {{
    color: {CORP['primary']}; background: #EFF6FF !important;
    border-bottom: 3px solid {CORP['primary']} !important;
  }}
  [data-testid="stSidebar"] {{
    background: {CORP['bg_card']}; border-right: 1px solid {CORP['border']};
  }}
  [data-testid="stSidebar"] .stMarkdown h2 {{
    font-size: 13px; text-transform: uppercase; letter-spacing: 0.8px;
    color: {CORP['text_sub']}; margin: 18px 0 6px;
  }}
  .weight-total {{
    font-size: 20px; font-weight: 700; text-align: center;
    padding: 10px; border-radius: 8px; margin-top: 8px;
  }}
  .weight-ok  {{ background:#D1FAE5; color:#065F46; }}
  .weight-err {{ background:#FEE2E2; color:#991B1B; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 0b. UTILITÁRIOS PLOTLY
# ══════════════════════════════════════════════════════════════════════════════
def corp_layout(**overrides) -> dict:
    base = dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=CORP["text_main"]),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white", bordercolor=CORP["border"],
            font_size=12, font_color=CORP["text_main"],
        ),
        legend=dict(
            orientation="h", y=1.06, x=0.5, xanchor="center",
            font_size=11, bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        margin=dict(t=70, b=40, l=50, r=20),
        xaxis=dict(showgrid=False, linecolor=CORP["border"], tickcolor=CORP["border"], zeroline=False),
        yaxis=dict(gridcolor="#F1F5F9", linecolor=CORP["border"], zeroline=False),
    )
    base.update(overrides)
    return base


def metric_card(label: str, value: str, accent_color: str = None, value_class: str = "") -> str:
    color = accent_color or CORP["primary"]
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-value {value_class}'>{value}</div>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-accent' style='background:{color}'></div>"
        f"</div>"
    )


def format_alloc_hover(weights_arr, asset_names, min_pct: float = 0.5, top_n: int = 10) -> str:
    """
    Formata a alocação (pesos) de uma carteira como uma string HTML multi-linha,
    pronta para uso em `customdata` + `hovertemplate` do Plotly.

    Mostra as posições relevantes (peso >= min_pct%) em ordem decrescente,
    limitadas a `top_n` linhas — o restante é agrupado em "+N outra(s)" para
    não poluir o tooltip quando há muitos ativos com peso pequeno.
    """
    pairs = sorted(zip(asset_names, weights_arr), key=lambda p: p[1], reverse=True)
    relevant = [(a, w) for a, w in pairs if w * 100 >= min_pct]
    if not relevant:
        return "sem posições relevantes"
    shown, rest = relevant[:top_n], relevant[top_n:]
    lines = [f"{a}: {w * 100:.1f}%" for a, w in shown]
    if rest:
        rest_pct = sum(w for _, w in rest)
        lines.append(f"+{len(rest)} outra(s): {rest_pct * 100:.1f}%")
    return "<br>".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 1. FUNÇÕES DE DADOS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def get_cdi_data(start_date, end_date):
    """Busca CDI mensal (Série 4391) na API do BCB — usado como benchmark e taxa livre de risco."""
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados?formato=json"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df.set_index("data", inplace=True)
        df["valor"] = df["valor"].astype(float) / 100.0
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        cdi_series = df.loc[mask, "valor"].resample("ME").last()
        cdi_series.name = "CDI"
        return cdi_series
    except Exception as e:
        st.error(f"Erro ao baixar dados do CDI (BCB): {e}")
        return pd.Series(dtype="float64")


@st.cache_data
def get_market_data(tickers, start_date, end_date):
    """Busca retornos mensais de ações via yFinance. Adiciona sufixo .SA automaticamente."""
    if not tickers:
        return pd.DataFrame()
    processed = []
    for t in tickers:
        t = t.strip().upper()
        processed.append(f"{t}.SA" if "." not in t and any(c.isdigit() for c in t) else t)
    try:
        data = yf.download(processed, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.DataFrame()
        if "Adj Close" in data.columns:
            prices = data["Adj Close"]
        elif "Close" in data.columns:
            prices = data["Close"]
        else:
            try:
                prices = data.xs("Adj Close", level=0, axis=1)
            except KeyError:
                prices = data.xs("Close", level=0, axis=1)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=processed[0])
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(-1)
        monthly_prices = prices.resample("ME").last()
        returns = monthly_prices.pct_change()
        returns.columns = [str(c).replace(".SA", "") for c in returns.columns]
        return returns
    except Exception as e:
        st.error(f"Erro no download (YFinance): {e}")
        return pd.DataFrame()


@st.cache_data
def get_benchmark_data(start_date, end_date):
    """Busca retornos mensais do Ibovespa via BOVA11."""
    try:
        ibov = yf.download("BOVA11.SA", start=start_date, end=end_date, progress=False)
        if ibov.empty:
            return pd.Series(dtype="float64")
        prices = ibov["Adj Close"] if "Adj Close" in ibov.columns else ibov.iloc[:, 0]
        returns = prices.resample("ME").last().pct_change().dropna()
        returns.name = "Ibovespa"
        return returns
    except Exception as e:
        st.error(f"Erro no benchmark: {e}")
        return pd.Series(dtype="float64")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FUNÇÕES DE CÁLCULO
# ══════════════════════════════════════════════════════════════════════════════
def calculate_portfolio_performance(returns_df, weights, initial_cap, monthly_contribution, rebalance_freq):
    returns_df = returns_df.dropna()
    available_assets = [c for c in returns_df.columns if c in weights and weights[c] > 0]
    if not available_assets:
        return None, None, None
    active_weights = np.array([weights[c] for c in available_assets])
    active_weights = active_weights / active_weights.sum()
    portfolio_pure_idx = [100.0]
    monthly_returns    = []
    portfolio_wealth   = [initial_cap]
    current_weights    = active_weights.copy()
    dates              = returns_df.index
    asset_returns_np   = returns_df[available_assets].values
    for i in range(len(dates)):
        r_t      = asset_returns_np[i]
        port_ret = np.dot(current_weights, r_t)
        monthly_returns.append(port_ret)
        portfolio_pure_idx.append(portfolio_pure_idx[-1] * (1 + port_ret))
        portfolio_wealth.append(portfolio_wealth[-1] * (1 + port_ret) + monthly_contribution)
        current_weights = current_weights * (1 + r_t) / (1 + port_ret)
        is_rebalance = (rebalance_freq == "Mensal") or (rebalance_freq == "Anual" and dates[i].month == 12)
        if is_rebalance:
            current_weights = active_weights.copy()
    return (
        pd.Series(portfolio_pure_idx[1:], index=dates),
        pd.Series(portfolio_wealth[1:], index=dates),
        pd.Series(monthly_returns, index=dates, name="Portfólio"),
    )


def create_monthly_heatmap(returns_series):
    df_ret = returns_series.to_frame(name="Retorno")
    df_ret["Ano"] = df_ret.index.year
    df_ret["Mes"] = df_ret.index.month
    pivot = df_ret.pivot(index="Ano", columns="Mes", values="Retorno")
    pivot["YTD"] = ((1 + pivot.fillna(0)).prod(axis=1) - 1)
    pivot.rename(columns={1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
                           7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}, inplace=True)
    return pivot


def run_walkforward_optimization(returns_df, rf_monthly_avg, window_months=6):
    n_rows, n_assets = returns_df.shape
    rf_ann = rf_monthly_avg * 12
    weights_list, window_info = [], []
    for start_idx in range(0, n_rows - window_months + 1, window_months):
        end_idx     = start_idx + window_months
        window_data = returns_df.iloc[start_idx:end_idx]
        if len(window_data) < window_months:
            continue
        mu_w    = window_data.mean().values * 12
        Sigma_w = window_data.cov().values   * 12

        def _neg_sharpe_wf(w, mu=mu_w, S=Sigma_w, rf=rf_ann):
            r = float(np.dot(w, mu))
            v = float(np.sqrt(np.maximum(w @ S @ w, 0.0)))
            return -(r - rf) / v if v > 1e-9 else 0.0

        w0     = np.full(n_assets, 1.0 / n_assets)
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        eq_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        try:
            res = minimize(_neg_sharpe_wf, w0, method="SLSQP",
                           bounds=bounds, constraints=[eq_sum],
                           options={"ftol": 1e-12, "maxiter": 1000})
            if res.success and np.isfinite(res.x).all():
                weights_list.append(np.clip(res.x, 0, 1))
                window_info.append((window_data.index[0], window_data.index[-1]))
        except Exception:
            continue
    return weights_list, window_info


def build_scenario_portfolio(weights_list, asset_names, method="median"):
    if not weights_list:
        return np.array([]), pd.DataFrame()
    weights_matrix = np.vstack(weights_list)
    df_windows     = pd.DataFrame(weights_matrix, columns=asset_names)
    w_raw    = np.median(weights_matrix, axis=0) if method == "median" else np.mean(weights_matrix, axis=0)
    total    = w_raw.sum()
    w_cen    = w_raw / total if total > 1e-9 else w_raw
    return w_cen, df_windows


def compute_scenario_metrics(returns_df, weights, rf_monthly_series):
    port_monthly = pd.Series(returns_df.values @ weights, index=returns_df.index)
    cdi_aligned  = rf_monthly_series.reindex(returns_df.index).fillna(0)
    excess  = port_monthly - cdi_aligned
    std_m   = port_monthly.std()
    vol_ann = std_m * np.sqrt(12)
    ret_ann = port_monthly.mean() * 12
    sharpe  = (excess.mean() / std_m) * np.sqrt(12) if std_m > 1e-9 else 0.0
    return sharpe, vol_ann, ret_ann


# ══════════════════════════════════════════════════════════════════════════════
# 3. SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0 4px'>
      <span style='font-size:28px'>📈</span><br>
      <span style='font-size:17px; font-weight:800; color:#0F172A; letter-spacing:-0.3px'>
        Equity Allocator Pro
      </span><br>
      <span style='font-size:10px; color:#94A3B8; text-transform:uppercase; letter-spacing:1px'>
        Análise de Ações Brasileiras
      </span>
    </div>
    <hr style='border:none; border-top:1px solid #E2E8F0; margin:12px 0'>
    """, unsafe_allow_html=True)

    # ── Período ───────────────────────────────────────────────────────────────
    st.markdown("## 📅 PERÍODO")
    min_date = datetime(2010, 1, 1)
    max_date = datetime.today()
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Início", datetime(2018, 1, 1), min_value=min_date, max_value=max_date)
    end_date   = col_d2.date_input("Fim",    max_date,             min_value=min_date, max_value=max_date)

    # ── Configurações ─────────────────────────────────────────────────────────
    st.markdown("## ⚙️ CONFIGURAÇÕES")
    rebalance_freq       = st.selectbox("Rebalanceamento", ["Mensal", "Anual"])
    investimento_inicial = st.number_input("Investimento Inicial (R$)", value=100_000.0, step=1_000.0, format="%.2f")
    aporte_mensal        = st.number_input("Aporte Mensal (R$)",        value=1_000.0,   step=100.0,   format="%.2f")

    # ── Carteira de Ações ─────────────────────────────────────────────────────
    st.markdown("## 📈 CARTEIRA DE AÇÕES")
    DEFAULT_STOCKS = "EGIE3, ITUB3, PSSA3, WEGE3, CXSE3, SBSP3, TAEE3, VIVT3, CPFE3, SAPR3, BBAS3, PRIO3, TOTS3, BPAC3, ALUP3, BMOB3"
    stocks_input = st.text_area("Tickers (separados por vírgula)", DEFAULT_STOCKS, height=100)
    stock_list   = [x.strip().upper() for x in stocks_input.split(",") if x.strip()]

    # ── Pesos Individuais ─────────────────────────────────────────────────────
    st.markdown("## ⚖️ PESOS DAS AÇÕES")
    equal_weights_toggle = st.toggle("Pesos Iguais (Automático)", value=True)

    stock_weight_inputs: dict[str, float] = {}

    if not equal_weights_toggle and stock_list:
        with st.expander("Ajustar pesos individualmente (%)", expanded=True):
            n_stocks  = len(stock_list)
            default_w = max(1, 100 // n_stocks)
            for ticker in stock_list:
                w = st.number_input(
                    ticker, min_value=0, max_value=100,
                    value=default_w, step=1, key=f"sw_{ticker}",
                )
                stock_weight_inputs[ticker] = float(w)
        total_w = sum(stock_weight_inputs.values())
        color_cls = "weight-ok" if abs(total_w - 100) < 0.5 else "weight-err"
        icon      = "✅" if abs(total_w - 100) < 0.5 else "⚠️"
        st.markdown(
            f"<div class='weight-total {color_cls}'>{icon} Total: {total_w:.0f}%</div>",
            unsafe_allow_html=True,
        )
        if abs(total_w - 100) > 0.5:
            st.caption("Pesos serão normalizados automaticamente para 100%.")
    else:
        n_stocks = len(stock_list)
        if n_stocks > 0:
            eq = 100.0 / n_stocks
            stock_weight_inputs = {t: eq for t in stock_list}
            st.markdown(
                f"<div class='badge badge-ok'>✅ {n_stocks} ação(ões) · {eq:.1f}% cada</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# 4. HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(
        "<h1 style='font-size:28px;font-weight:800;color:#0F172A;margin-bottom:0'>"
        "📈 Equity Allocator Pro</h1>"
        "<p style='color:#64748B;margin-top:2px;font-size:13px'>"
        f"Período: <b>{start_date.strftime('%d/%m/%Y')}</b> → <b>{end_date.strftime('%d/%m/%Y')}</b>"
        f" | Rebalanceamento: <b>{rebalance_freq}</b>"
        f" | <b>{len(stock_list)}</b> ação(ões) na carteira</p>",
        unsafe_allow_html=True,
    )
with h2:
    badge_cls = "badge-ok" if stock_list else "badge-warn"
    badge_txt = f"📈 {len(stock_list)} ações" if stock_list else "Nenhuma ação"
    st.markdown(f"<div class='badge {badge_cls}' style='margin-top:20px'>{badge_txt}</div>", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid #E2E8F0;margin:4px 0 16px'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5. CARREGAMENTO DE DADOS
# ══════════════════════════════════════════════════════════════════════════════
if not stock_list:
    st.warning("⚠️ Adicione pelo menos uma ação na barra lateral para iniciar.")
    st.stop()

with st.spinner("📡 Carregando cotações, Ibovespa e CDI..."):
    api_start = pd.to_datetime(start_date) - pd.Timedelta(days=45)
    df_stocks = get_market_data(stock_list, api_start, end_date)
    ibov_ret  = get_benchmark_data(api_start, end_date)
    cdi_ret   = get_cdi_data(api_start, end_date)

if df_stocks.empty:
    st.error("❌ Nenhum dado retornado para as ações informadas. Verifique os tickers e sua conexão.")
    st.stop()

# ── Master DataFrame (com fillna para cálculos de portfólio) ─────────────────
mask = (df_stocks.index >= pd.to_datetime(start_date)) & (df_stocks.index <= pd.to_datetime(end_date))
master_df  = df_stocks.loc[mask].fillna(0)   # para backtest / fronteira eficiente
corr_df    = df_stocks.loc[mask]              # sem fillna — para correlações reais

ibov_ret       = ibov_ret.reindex(master_df.index).fillna(0)
cdi_ret_series = cdi_ret.reindex(master_df.index).fillna(0)

# ── Dicionário de pesos normalizados ─────────────────────────────────────────
raw_total = sum(stock_weight_inputs.get(t, 0) for t in stock_list)
if raw_total > 0:
    weights = {
        t: stock_weight_inputs.get(t, 0) / raw_total
        for t in stock_list
        if t in master_df.columns
    }
else:
    available = [t for t in stock_list if t in master_df.columns]
    weights   = {t: 1.0 / len(available) for t in available} if available else {}

# Identifica ações efetivamente carregadas
loaded_stocks = [t for t in stock_list if t in master_df.columns]
missing_stocks = [t for t in stock_list if t not in master_df.columns]
if missing_stocks:
    st.warning(f"⚠️ Tickers não encontrados e ignorados: **{', '.join(missing_stocks)}**")

# ══════════════════════════════════════════════════════════════════════════════
# 6. BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
port_pure, port_wealth, port_ret = calculate_portfolio_performance(
    master_df, weights, investimento_inicial, aporte_mensal, rebalance_freq
)

if port_ret is None:
    st.warning("⚠️ Nenhuma ação com peso > 0. Ajuste a carteira na barra lateral.")
    st.stop()

# ── Métricas principais ───────────────────────────────────────────────────────
cdi_ret_series = cdi_ret.reindex(port_ret.index).fillna(0)
cdi_accum      = (1 + cdi_ret_series).cumprod() * 100
ibov_accum     = (1 + ibov_ret.reindex(port_ret.index).fillna(0)).cumprod() * 100
ibov_aligned   = ibov_ret.reindex(port_ret.index).fillna(0)

total_ret = (port_pure.iloc[-1] / 100) - 1
years     = len(port_ret) / 12
cagr      = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
vol       = port_ret.std() * np.sqrt(12)
excess_r  = port_ret - cdi_ret_series
sharpe    = (excess_r.mean() / port_ret.std()) * np.sqrt(12) if port_ret.std() > 0 else 0
cum_ret   = (1 + port_ret).cumprod()
peak      = cum_ret.cummax()
dd_series = (cum_ret - peak) / peak
max_dd    = dd_series.min()


# ══════════════════════════════════════════════════════════════════════════════
# 7. DASHBOARD — METRIC CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📐 Performance Overview</div>", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, label, value, accent, cls in [
    (c1, "Retorno Total",   f"{total_ret:.1%}", CORP["primary"],  "positive" if total_ret >= 0 else "negative"),
    (c2, "CAGR (a.a.)",     f"{cagr:.1%}",       CORP["success"],  "positive" if cagr >= 0 else "negative"),
    (c3, "Volatilidade",    f"{vol:.1%}",          CORP["warning"],  ""),
    (c4, "Sharpe vs CDI",   f"{sharpe:.2f}",       CORP["primary"],  "positive" if sharpe >= 1 else ("negative" if sharpe < 0 else "")),
    (c5, "Max Drawdown",    f"{max_dd:.1%}",       CORP["danger"],   "negative"),
    (c6, "Período (meses)", f"{len(port_ret)}",    CORP["neutral"],  ""),
]:
    col.markdown(metric_card(label, value, accent, cls), unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_perf, tab_corr, tab_risk, tab_month, tab_patr, tab_proj, tab_ef = st.tabs([
    "📈 Rentabilidade",
    "🔗 Correlação",
    "🛡️ Risco",
    "📅 Retornos Mensais",
    "💰 Patrimônio",
    "🔮 Projeções",
    "🎯 Fronteira Eficiente",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RENTABILIDADE
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown("<div class='section-header'>Evolução Acumulada — Portfólio vs Benchmarks (Base 100)</div>",
                unsafe_allow_html=True)

    fig = go.Figure()
    for name, y, color, width, dash in [
        ("Portfólio",      port_pure.values,  CORP["primary"],  3.0, "solid"),
        ("Ibovespa",       ibov_accum.values, CORP["warning"],  2.0, "dot"),
        ("CDI Real (BCB)", cdi_accum.values,  CORP["success"],  1.8, "dash"),
    ]:
        fig.add_trace(go.Scatter(
            x=port_pure.index, y=y, name=name, mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"<b>{name}</b><br>%{{x|%b/%Y}}: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(**corp_layout(
        title=dict(text="Comparativo de Rentabilidade Acumulada", font_size=14),
        yaxis=dict(title="Índice (Base 100)", gridcolor="#F1F5F9", zeroline=False),
        height=420,
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ── Desempenho Individual das Ações ───────────────────────────────────────
    with st.expander("📊 Desempenho Individual das Ações (Base 100)", expanded=False):
        fig_ind = go.Figure()
        for idx, ticker in enumerate(loaded_stocks):
            if ticker not in master_df.columns:
                continue
            s_ret  = master_df[ticker].replace(0, np.nan).dropna()
            s_accum = (1 + s_ret).cumprod() * 100
            clr    = STOCK_COLORS[idx % len(STOCK_COLORS)]
            fig_ind.add_trace(go.Scatter(
                x=s_accum.index, y=s_accum.values,
                name=ticker, mode="lines",
                line=dict(color=clr, width=1.6),
                hovertemplate=f"<b>{ticker}</b><br>%{{x|%b/%Y}}: %{{y:.1f}}<extra></extra>",
            ))
        # Portfólio em destaque
        fig_ind.add_trace(go.Scatter(
            x=port_pure.index, y=port_pure.values,
            name="◉ Portfólio", mode="lines",
            line=dict(color=CORP["text_main"], width=3.5),
            hovertemplate="<b>Portfólio</b><br>%{x|%b/%Y}: %{y:.1f}<extra></extra>",
        ))
        fig_ind.update_layout(**corp_layout(
            title=dict(text="Retorno Acumulado por Ação (Base 100)", font_size=13),
            yaxis=dict(title="Índice (Base 100)", gridcolor="#F1F5F9"),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font_size=10),
            height=480, margin=dict(t=60, b=120),
        ))
        st.plotly_chart(fig_ind, use_container_width=True)

    # ── Tabela de Rentabilidade por Período ───────────────────────────────────
    st.markdown("<div class='section-header'>Rentabilidade por Período</div>", unsafe_allow_html=True)
    periods = {"12M": 12, "24M": 24, "36M": 36, "48M": 48, "Início": len(port_ret)}
    rows = {}
    for label_p, n in periods.items():
        if len(port_ret) >= n:
            p_ret = (1 + port_ret.tail(n)).prod() - 1
            i_ret = (1 + ibov_aligned.tail(n)).prod() - 1
            c_ret = (1 + cdi_ret_series.tail(n)).prod() - 1
            rows[label_p] = {
                "Portfólio":     f"{p_ret:.2%}",
                "Ibovespa":      f"{i_ret:.2%}",
                "CDI":           f"{c_ret:.2%}",
                "Alpha vs CDI":  f"{p_ret - c_ret:+.2%}",
                "Alpha vs IBOV": f"{p_ret - i_ret:+.2%}",
            }
    if rows:
        st.dataframe(
            pd.DataFrame(rows).T
              .style.set_properties(**{"text-align": "center"})
              .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
            use_container_width=True,
        )

    # ── Tabela de Retorno Acumulado por Ação ──────────────────────────────────
    st.markdown("<div class='section-header'>Retorno Acumulado por Ação (Período Completo)</div>",
                unsafe_allow_html=True)
    stock_metrics = []
    for ticker in loaded_stocks:
        if ticker not in master_df.columns:
            continue
        s = master_df[ticker].replace(0, np.nan).dropna()
        if len(s) < 2:
            continue
        total_r  = (1 + s).prod() - 1
        y_s      = len(s) / 12
        cagr_s   = (1 + total_r) ** (1 / y_s) - 1 if y_s > 0 else 0
        vol_s    = s.std() * np.sqrt(12)
        dd_s_cum = (1 + s).cumprod()
        dd_s     = ((dd_s_cum - dd_s_cum.cummax()) / dd_s_cum.cummax()).min()
        stock_metrics.append({
            "Ticker":          ticker,
            "Retorno Total":   f"{total_r:.1%}",
            "CAGR (a.a.)":     f"{cagr_s:.1%}",
            "Volatilidade":    f"{vol_s:.1%}",
            "Max Drawdown":    f"{dd_s:.1%}",
            "Peso Carteira":   f"{weights.get(ticker, 0):.1%}",
        })
    if stock_metrics:
        st.dataframe(
            pd.DataFrame(stock_metrics).set_index("Ticker")
              .style.set_properties(**{"text-align": "center"})
              .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CORRELAÇÃO  ★ NOVA ★
# ══════════════════════════════════════════════════════════════════════════════
with tab_corr:

    # ── Preparação dos dados de correlação (sem fillna) ───────────────────────
    min_obs     = max(6, int(len(corr_df) * 0.25))
    valid_cols  = [c for c in corr_df.columns if corr_df[c].notna().sum() >= min_obs]
    corr_clean  = corr_df[valid_cols]

    if len(valid_cols) < 2:
        st.warning("⚠️ São necessárias ao menos 2 ações com dados suficientes para a análise de correlação.")
        st.stop()

    corr_matrix = corr_clean.corr()

    # ── Heatmap de Correlação ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔗 Matriz de Correlação — Retornos Mensais</div>",
                unsafe_allow_html=True)

    n_valid     = len(valid_cols)
    show_text   = n_valid <= 14          # anotações só se couber
    cell_size   = max(30, min(60, 600 // n_valid))
    hmap_height = max(420, cell_size * n_valid + 120)

    corr_vals   = corr_matrix.values
    fig_hmap    = go.Figure(data=go.Heatmap(
        z=corr_vals,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_vals, 2) if show_text else None,
        texttemplate="%{text}" if show_text else None,
        textfont=dict(size=9),
        hoverongaps=False,
        colorbar=dict(title="Corr", len=0.8, thickness=14),
        hovertemplate="<b>%{y} × %{x}</b><br>Correlação: %{z:.3f}<extra></extra>",
    ))
    fig_hmap.update_layout(
        **corp_layout(
            xaxis=dict(side="bottom", tickangle=-40, showgrid=False, linecolor=CORP["border"]),
            yaxis=dict(showgrid=False, linecolor=CORP["border"]),
            height=hmap_height,
            margin=dict(t=60, b=80, l=80, r=60),
        )
    )
    st.plotly_chart(fig_hmap, use_container_width=True)

    # ── Tabelas de pares + métricas de diversificação ─────────────────────────
    # Extrai pares únicos
    pairs = []
    cols  = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({"Par": f"{cols[i]} × {cols[j]}", "Correlação": corr_matrix.iloc[i, j]})
    df_pairs = pd.DataFrame(pairs).sort_values("Correlação", ascending=False).reset_index(drop=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("<div class='section-header'>🔴 Pares Mais Correlacionados</div>", unsafe_allow_html=True)
        st.dataframe(
            df_pairs.head(8).style.format({"Correlação": "{:.3f}"})
              .background_gradient(cmap="Reds", subset=["Correlação"], vmin=0, vmax=1),
            use_container_width=True, hide_index=True,
        )
    with col_c2:
        st.markdown("<div class='section-header'>🟢 Pares Mais Diversificantes</div>", unsafe_allow_html=True)
        st.dataframe(
            df_pairs.tail(8).iloc[::-1].reset_index(drop=True)
              .style.format({"Correlação": "{:.3f}"})
              .background_gradient(cmap="Blues_r", subset=["Correlação"], vmin=-1, vmax=0.5),
            use_container_width=True, hide_index=True,
        )

    # ── Métricas de Diversificação ────────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Métricas de Diversificação do Portfólio</div>",
                unsafe_allow_html=True)

    avg_pairwise_corr = df_pairs["Correlação"].mean() if not df_pairs.empty else 0.0

    # N Efetivo (Herfindahl inverso)
    w_arr  = np.array([weights.get(t, 0) for t in valid_cols])
    if w_arr.sum() > 0:
        w_arr  = w_arr / w_arr.sum()
        n_efetivo = 1.0 / np.sum(w_arr ** 2) if np.sum(w_arr ** 2) > 0 else 1.0
    else:
        n_efetivo = float(n_valid)

    # Correlação média ponderada do portfólio
    n_eff_int = len(valid_cols)
    w_norm    = np.ones(n_eff_int) / n_eff_int if w_arr.sum() == 0 else w_arr
    port_avg_corr = 0.0
    for i in range(n_eff_int):
        for j in range(n_eff_int):
            if i != j:
                port_avg_corr += w_norm[i] * w_norm[j] * corr_matrix.iloc[i, j]

    # Beta médio ponderado vs Ibovespa
    ibov_for_beta = ibov_ret.reindex(corr_clean.index)
    betas = {}
    for ticker in valid_cols:
        s = corr_clean[ticker].dropna()
        common = s.index.intersection(ibov_for_beta.dropna().index)
        if len(common) >= 12:
            x = ibov_for_beta.loc[common].values
            y = s.loc[common].values
            slope, _, _, _, _ = stats.linregress(x, y)
            betas[ticker] = slope
    if betas:
        beta_port = sum(weights.get(t, 0) * b for t, b in betas.items())
        total_beta_w = sum(weights.get(t, 0) for t in betas)
        beta_port = beta_port / total_beta_w if total_beta_w > 0 else np.nan
    else:
        beta_port = np.nan

    m1, m2, m3, m4 = st.columns(4)
    for col_m, lbl, val, ac in [
        (m1, "Corr. Média (Todos Pares)",   f"{avg_pairwise_corr:.3f}", CORP["primary"]),
        (m2, "Corr. Ponderada do Portfólio", f"{port_avg_corr:.3f}",   CORP["warning"]),
        (m3, "N Efetivo de Apostas",         f"{n_efetivo:.1f}",        CORP["success"]),
        (m4, "Beta do Portfólio (vs IBOV)",  f"{beta_port:.2f}" if not np.isnan(beta_port) else "N/A", CORP["purple"]),
    ]:
        col_m.markdown(metric_card(lbl, val, ac), unsafe_allow_html=True)

    st.caption(
        "**Corr. Média** = média simples de todos os pares. "
        "**Corr. Ponderada** = correlação média ponderada pelos pesos. "
        "**N Efetivo** = 1/Σwᵢ² (quanto mais alto, mais diversificado). "
        "**Beta** = sensibilidade estimada ao Ibovespa."
    )

    # ── Análise de Par ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔍 Análise de Par — Correlação Móvel & Dispersão</div>",
                unsafe_allow_html=True)

    if len(valid_cols) >= 2:
        col_sel1, col_sel2, col_window = st.columns([2, 2, 1])
        with col_sel1:
            ativo_a = st.selectbox("Ação A", valid_cols, index=0, key="corr_a")
        with col_sel2:
            default_b = 1 if valid_cols[0] == ativo_a and len(valid_cols) > 1 else 0
            remaining = [v for v in valid_cols if v != ativo_a]
            ativo_b   = st.selectbox("Ação B", remaining, index=0, key="corr_b")
        with col_window:
            roll_win = st.selectbox("Janela", [6, 12, 24], index=1, key="roll_win")

        if ativo_a != ativo_b:
            s_a     = corr_clean[ativo_a]
            s_b     = corr_clean[ativo_b]
            common  = s_a.dropna().index.intersection(s_b.dropna().index)
            s_a_c   = s_a.loc[common]
            s_b_c   = s_b.loc[common]
            roll_c  = s_a_c.rolling(roll_win).corr(s_b_c)

            # Correlação global e R² do par
            if len(common) >= 6:
                slope_ab, intercept_ab, r_ab, _, _ = stats.linregress(s_a_c.values, s_b_c.values)
            else:
                slope_ab = intercept_ab = r_ab = np.nan

            col_rc, col_sc = st.columns(2)
            with col_rc:
                fig_rc = go.Figure()
                fig_rc.add_hline(y=0,    line_dash="dash", line_color=CORP["neutral"], line_width=1)
                fig_rc.add_hline(y=0.5,  line_dash="dot",  line_color=CORP["warning"], line_width=1, annotation_text="0.5")
                fig_rc.add_hline(y=-0.5, line_dash="dot",  line_color=CORP["success"], line_width=1, annotation_text="-0.5")
                fig_rc.add_trace(go.Scatter(
                    x=roll_c.index, y=roll_c.values,
                    mode="lines", name=f"Corr. Móvel {roll_win}M",
                    line=dict(color=CORP["primary"], width=2),
                    fill="tozeroy", fillcolor="rgba(26,86,219,0.08)",
                    hovertemplate="%{x|%b/%Y}: %{y:.3f}<extra></extra>",
                ))
                fig_rc.update_layout(**corp_layout(
                    title=dict(text=f"Correlação Móvel {roll_win}M — {ativo_a} × {ativo_b}", font_size=13),
                    yaxis=dict(title="Correlação", range=[-1.1, 1.1], gridcolor="#F1F5F9"),
                    height=340, margin=dict(t=55, b=35, l=50, r=20),
                    legend=dict(orientation="h", y=1.12),
                ))
                st.plotly_chart(fig_rc, use_container_width=True)

            with col_sc:
                if len(common) >= 6:
                    x_line  = np.linspace(s_a_c.min(), s_a_c.max(), 100)
                    y_line  = intercept_ab + slope_ab * x_line
                    fig_sc  = go.Figure()
                    fig_sc.add_trace(go.Scatter(
                        x=s_a_c.values, y=s_b_c.values,
                        mode="markers",
                        marker=dict(color=CORP["primary"], size=6, opacity=0.65,
                                    line=dict(color="white", width=0.5)),
                        hovertemplate=f"{ativo_a}: %{{x:.2%}}<br>{ativo_b}: %{{y:.2%}}<extra></extra>",
                        name="Observações",
                    ))
                    fig_sc.add_trace(go.Scatter(
                        x=x_line, y=y_line, mode="lines",
                        line=dict(color=CORP["danger"], width=2, dash="dash"),
                        name=f"Regressão  R²={r_ab**2:.3f}",
                    ))
                    fig_sc.add_annotation(
                        x=0.05, y=0.95, xref="paper", yref="paper",
                        text=f"r = {r_ab:.3f} | R² = {r_ab**2:.3f}",
                        showarrow=False, bgcolor="white",
                        bordercolor=CORP["border"], borderwidth=1,
                        font=dict(size=11, color=CORP["text_main"]),
                    )
                    fig_sc.update_layout(**corp_layout(
                        title=dict(text=f"Dispersão dos Retornos — {ativo_a} × {ativo_b}", font_size=13),
                        xaxis=dict(title=f"Retorno Mensal {ativo_a}", tickformat=".1%", showgrid=False),
                        yaxis=dict(title=f"Retorno Mensal {ativo_b}", tickformat=".1%", gridcolor="#F1F5F9"),
                        hovermode="closest",
                        height=340, margin=dict(t=55, b=45, l=60, r=20),
                        legend=dict(orientation="h", y=1.12),
                    ))
                    st.plotly_chart(fig_sc, use_container_width=True)

            # Estatísticas do par
            if len(common) >= 6:
                corr_atual     = roll_c.dropna().iloc[-1] if not roll_c.dropna().empty else np.nan
                corr_atual_str = f"{corr_atual:.3f}" if not np.isnan(corr_atual) else "N/A"
                st.info(
                    f"**{ativo_a} × {ativo_b}** | "
                    f"Corr. Global: **{r_ab:.3f}** | "
                    f"Corr. Móvel Atual ({roll_win}M): **{corr_atual_str}** | "
                    f"R\u00b2 = **{r_ab ** 2:.3f}** | "
                    f"Beta ({ativo_b} ~ {ativo_a}): **{slope_ab:.3f}**"
                )

    # ── Mapa de calor de correlação com Ibovespa ──────────────────────────────
    st.markdown("<div class='section-header'>📉 Correlação de Cada Ação com o Ibovespa</div>",
                unsafe_allow_html=True)

    ibov_corrs = {}
    ibov_betas_all = {}
    for ticker in valid_cols:
        s      = corr_clean[ticker].dropna()
        common = s.index.intersection(ibov_for_beta.dropna().index)
        if len(common) >= 12:
            corr_v = s.loc[common].corr(ibov_for_beta.loc[common])
            ibov_corrs[ticker] = corr_v
            sl, ic, _, _, _ = stats.linregress(ibov_for_beta.loc[common].values, s.loc[common].values)
            ibov_betas_all[ticker] = sl

    if ibov_corrs:
        df_ibov_corr = pd.DataFrame({
            "Ticker":          list(ibov_corrs.keys()),
            "Corr. c/ IBOV":  [ibov_corrs[t] for t in ibov_corrs],
            "Beta vs IBOV":    [ibov_betas_all.get(t, np.nan) for t in ibov_corrs],
        }).sort_values("Corr. c/ IBOV", ascending=True)

        fig_bar_corr = go.Figure()
        fig_bar_corr.add_trace(go.Bar(
            x=df_ibov_corr["Ticker"],
            y=df_ibov_corr["Corr. c/ IBOV"],
            marker_color=[CORP["success"] if v < 0.6 else CORP["warning"] if v < 0.8 else CORP["danger"]
                          for v in df_ibov_corr["Corr. c/ IBOV"]],
            text=[f"{v:.2f}" for v in df_ibov_corr["Corr. c/ IBOV"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Corr: %{y:.3f}<extra></extra>",
        ))
        fig_bar_corr.add_hline(y=0.7, line_dash="dash", line_color=CORP["warning"],
                                annotation_text="Alta Corr. (0.7)", line_width=1.5)
        fig_bar_corr.update_layout(**corp_layout(
            title=dict(text="Correlação de Cada Ação com o Ibovespa (BOVA11)", font_size=13),
            xaxis=dict(title="Ação", showgrid=False),
            yaxis=dict(title="Correlação", range=[0, 1.15], gridcolor="#F1F5F9"),
            showlegend=False,
            height=350, margin=dict(t=55, b=50, l=50, r=20),
        ))
        st.plotly_chart(fig_bar_corr, use_container_width=True)

        st.dataframe(
            df_ibov_corr.sort_values("Corr. c/ IBOV", ascending=False)
              .style.format({"Corr. c/ IBOV": "{:.3f}", "Beta vs IBOV": "{:.3f}"})
              .background_gradient(cmap="RdYlGn_r", subset=["Corr. c/ IBOV"], vmin=0, vmax=1)
              .set_properties(**{"text-align": "center"})
              .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISCO
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("<div class='section-header'>Drawdown Submarino</div>", unsafe_allow_html=True)
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series.values * 100,
            mode="lines", fill="tozeroy", name="Drawdown",
            line=dict(color=CORP["danger"], width=1.5),
            fillcolor="rgba(224,36,36,0.12)",
            hovertemplate="%{x|%b/%Y}: %{y:.2f}%<extra></extra>",
        ))
        fig_dd.update_layout(**corp_layout(
            yaxis=dict(title="Drawdown (%)", gridcolor="#F1F5F9", zeroline=False),
            height=300, margin=dict(t=40, b=30, l=50, r=10),
            legend=dict(orientation="h", y=1.1),
        ))
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_r2:
        st.markdown("<div class='section-header'>Volatilidade Móvel (12M)</div>", unsafe_allow_html=True)
        rolling_vol = port_ret.rolling(12).std() * np.sqrt(12) * 100
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol.values,
            mode="lines", name="Vol. 12M",
            line=dict(color=CORP["warning"], width=2),
            hovertemplate="%{x|%b/%Y}: %{y:.2f}%<extra></extra>",
        ))
        fig_vol.update_layout(**corp_layout(
            yaxis=dict(title="Volatilidade (%)", gridcolor="#F1F5F9", zeroline=False),
            height=300, margin=dict(t=40, b=30, l=50, r=10),
            legend=dict(orientation="h", y=1.1),
        ))
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("<div class='section-header'>Estatísticas de Risco Detalhadas</div>", unsafe_allow_html=True)
    months_pos  = (port_ret > 0).sum()
    months_neg  = (port_ret < 0).sum()
    calmar      = cagr / abs(max_dd) if max_dd != 0 else 0
    sortino_exc = port_ret[port_ret < 0].std() * np.sqrt(12)
    sortino     = (cagr - cdi_ret_series.mean() * 12) / sortino_exc if sortino_exc > 0 else 0

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    for col_s, lbl, val, ac, cls in [
        (s1, "Meses Positivos", f"{months_pos} ({months_pos/len(port_ret):.0%})", CORP["success"], ""),
        (s2, "Meses Negativos", f"{months_neg} ({months_neg/len(port_ret):.0%})", CORP["danger"],  ""),
        (s3, "Melhor Mês",      f"{port_ret.max():.2%}",  CORP["success"], "positive"),
        (s4, "Pior Mês",        f"{port_ret.min():.2%}",  CORP["danger"],  "negative"),
        (s5, "Ratio de Calmar", f"{calmar:.2f}",           CORP["primary"], ""),
        (s6, "Ratio de Sortino",f"{sortino:.2f}",          CORP["purple"],  ""),
    ]:
        col_s.markdown(metric_card(lbl, val, ac, cls), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RETORNOS MENSAIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_month:
    st.markdown("<div class='section-header'>Tabela de Rentabilidade — Heatmap</div>", unsafe_allow_html=True)
    heatmap_data = create_monthly_heatmap(port_ret)
    st.dataframe(
        heatmap_data.style
          .format("{:.2%}")
          .background_gradient(cmap="RdYlGn", vmin=-0.05, vmax=0.05, axis=None)
          .highlight_null(color="white"),
        use_container_width=True, height=400,
    )

    st.markdown("<div class='section-header'>Índice de Sharpe por Janela</div>", unsafe_allow_html=True)
    sharpe_periods = {"12M": 12, "24M": 24, "48M": 48, "60M": 60, "Início": len(port_ret)}
    sharpe_results = {}
    for lbl_s, months in sharpe_periods.items():
        if len(port_ret) >= months:
            sub_p = port_ret.tail(months)
            sub_c = cdi_ret_series.tail(months)
            v     = sub_p.std()
            sharpe_results[lbl_s] = ((sub_p - sub_c).mean() / v) * np.sqrt(12) if v > 0 else 0.0
        else:
            sharpe_results[lbl_s] = None
    df_sharpe = pd.DataFrame([sharpe_results], index=["Índice de Sharpe"])
    st.dataframe(
        df_sharpe.style.format("{:.2f}", na_rep="—").background_gradient(cmap="Blues", axis=1, vmin=0, vmax=2),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PATRIMÔNIO
# ══════════════════════════════════════════════════════════════════════════════
with tab_patr:
    st.markdown("<div class='section-header'>Crescimento Patrimonial</div>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns([3, 1])

    with col_p1:
        fig_wealth = go.Figure()
        fig_wealth.add_trace(go.Scatter(
            x=port_wealth.index, y=port_wealth.values,
            mode="lines", name="Patrimônio",
            fill="tozeroy",
            line=dict(color=CORP["success"], width=2.5),
            fillcolor="rgba(14,159,110,0.10)",
            hovertemplate="%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
        ))
        fig_wealth.update_layout(**corp_layout(
            title=dict(text="Crescimento Patrimonial (Cotas + Aportes)", font_size=14),
            yaxis=dict(title="Saldo (R$)", tickformat=",.0f", gridcolor="#F1F5F9"),
            height=380,
        ))
        st.plotly_chart(fig_wealth, use_container_width=True)

    with col_p2:
        final_val      = port_wealth.iloc[-1]
        total_invested = investimento_inicial + (aporte_mensal * len(port_ret))
        profit_loss    = final_val - total_invested
        roi_pct        = final_val / total_invested - 1
        st.metric("Saldo Final",      f"R$ {final_val:,.0f}")
        st.metric("Total Investido",  f"R$ {total_invested:,.0f}")
        st.metric("Lucro / Prejuízo", f"R$ {profit_loss:,.0f}", delta=f"{roi_pct:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PROJEÇÕES (MONTE CARLO)
# ══════════════════════════════════════════════════════════════════════════════
with tab_proj:
    st.markdown("<div class='section-header'>Projeção de Cenários — Próximos 36 Meses</div>", unsafe_allow_html=True)

    mu_mc = port_ret.mean()
    sg_mc = port_ret.std()
    N_M, N_S = 36, 20_000
    saldo_t0  = port_wealth.iloc[-1]
    last_date = port_wealth.index[-1]

    np.random.seed(42)
    paths         = np.empty((N_S, N_M + 1))
    paths[:, 0]   = saldo_t0
    rand_r        = np.random.normal(mu_mc, sg_mc, size=(N_S, N_M))
    for t in range(1, N_M + 1):
        paths[:, t] = paths[:, t - 1] * (1 + rand_r[:, t - 1]) + aporte_mensal

    p_ot = np.percentile(paths, 95, axis=0)
    p_ne = np.percentile(paths, 50, axis=0)
    p_pe = np.percentile(paths, 5,  axis=0)

    hist_tail    = port_wealth.tail(12)
    future_dates = pd.date_range(start=last_date, periods=N_M + 1, freq="ME")[1:]
    proj_dates   = [last_date] + list(future_dates)

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(
        x=list(proj_dates) + list(reversed(proj_dates)),
        y=list(p_ot) + list(reversed(p_pe)),
        fill="toself", fillcolor="rgba(26,86,219,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True, name="Intervalo P5–P95", hoverinfo="skip",
    ))
    for label_p, y_p, color_p, dash_p, tmpl in [
        ("Histórico Real",    hist_tail.values,  CORP["text_main"], "solid", "<b>Histórico</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>"),
        ("Otimista (P95)",    p_ot,               CORP["success"],   "dash",  "<b>Otimista</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>"),
        ("Neutro (P50)",      p_ne,               CORP["primary"],   "dot",   "<b>Neutro</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>"),
        ("Pessimista (P5)",   p_pe,               CORP["danger"],    "dash",  "<b>Pessimista</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>"),
    ]:
        x_vals = hist_tail.index if label_p == "Histórico Real" else proj_dates
        y_vals = hist_tail.values if label_p == "Histórico Real" else y_p
        fig_proj.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="lines", name=label_p,
            line=dict(color=color_p, width=2.5 if label_p == "Histórico Real" else 2.0, dash=dash_p),
            hovertemplate=tmpl,
        ))
    fig_proj.add_vline(x=last_date, line_width=1.2, line_dash="dot", line_color=CORP["neutral"])
    fig_proj.update_layout(**corp_layout(
        title=dict(text=f"Monte Carlo — {N_S:,} simulações | µ={mu_mc:.2%}/mês | σ={sg_mc:.2%}/mês | Aporte R$ {aporte_mensal:,.0f}/mês", font_size=12),
        yaxis=dict(title="Saldo (R$)", tickformat=",.0f", gridcolor="#F1F5F9"),
        height=430, margin=dict(t=80, b=40),
    ))
    st.plotly_chart(fig_proj, use_container_width=True)

    st.markdown("<div class='section-header'>Saldo Final Projetado em 36 Meses</div>", unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    pc1.markdown(metric_card("🟢 Otimista (P95)", f"R$ {p_ot[-1]:,.0f}", CORP["success"]), unsafe_allow_html=True)
    pc2.markdown(metric_card("🔵 Neutro (P50)",   f"R$ {p_ne[-1]:,.0f}", CORP["primary"]), unsafe_allow_html=True)
    pc3.markdown(metric_card("🔴 Pessimista (P5)", f"R$ {p_pe[-1]:,.0f}", CORP["danger"]),  unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — FRONTEIRA EFICIENTE + WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_ef:
    st.markdown("<div class='section-header'>🎯 Fronteira Eficiente de Markowitz</div>", unsafe_allow_html=True)

    active_assets = [a for a, w in weights.items() if w > 0 and a in master_df.columns]

    if len(active_assets) < 2:
        st.warning("⚠️ A Fronteira Eficiente requer pelo menos **2 ações com peso > 0**.")
    else:
        returns_ef = master_df[active_assets].replace(0, np.nan).dropna(how="all").fillna(0)
        mu_vec     = returns_ef.mean() * 12
        Sigma      = returns_ef.cov()  * 12
        rf_rate    = cdi_ret_series.mean() * 12
        n_assets   = len(active_assets)
        Sigma_np   = Sigma.values
        mu_np      = mu_vec.values

        def port_return(w): return float(np.dot(w, mu_np))
        def port_vol(w):    return float(np.sqrt(w @ Sigma_np @ w))
        def neg_sharpe(w):
            r, v = port_return(w), port_vol(w)
            return -(r - rf_rate) / v if v > 1e-9 else 0.0

        w0     = np.full(n_assets, 1.0 / n_assets)
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        eq_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res_minvol = minimize(port_vol,   w0, method="SLSQP", bounds=bounds, constraints=[eq_sum], options={"ftol": 1e-12, "maxiter": 1000})
        res_maxsh  = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=[eq_sum], options={"ftol": 1e-12, "maxiter": 1000})
        w_minvol, w_maxsh = res_minvol.x, res_maxsh.x

        ret_minvol, vol_minvol = port_return(w_minvol), port_vol(w_minvol)
        ret_maxsh,  vol_maxsh  = port_return(w_maxsh),  port_vol(w_maxsh)
        shrp_minvol = (ret_minvol - rf_rate) / vol_minvol if vol_minvol > 1e-9 else 0.0
        shrp_maxsh  = (ret_maxsh  - rf_rate) / vol_maxsh  if vol_maxsh  > 1e-9 else 0.0

        raw_w_cur = np.array([weights[a] for a in active_assets], dtype=float)
        w_cur     = raw_w_cur / raw_w_cur.sum()
        ret_cur, vol_cur = port_return(w_cur), port_vol(w_cur)
        shrp_cur  = (ret_cur - rf_rate) / vol_cur if vol_cur > 1e-9 else 0.0

        # Fronteira (guarda também os pesos ótimos de cada ponto, para o hover)
        target_rets = np.linspace(ret_minvol, mu_np.max() * 1.05, 120)
        frontier_vols, frontier_rets, frontier_weights = [], [], []
        for tgt in target_rets:
            cons  = [eq_sum, {"type": "eq", "fun": lambda w, t=tgt: port_return(w) - t}]
            res_f = minimize(port_vol, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-12, "maxiter": 800})
            if res_f.success and res_f.fun < 2.0:
                frontier_vols.append(res_f.fun)
                frontier_rets.append(tgt)
                frontier_weights.append(np.clip(res_f.x, 0, 1))

        indiv_vols = [float(np.sqrt(Sigma_np[i, i])) for i in range(n_assets)]
        indiv_rets = [float(mu_np[i])                 for i in range(n_assets)]

        # Texto HTML (uma string por ponto) com a alocação daquele ponto da fronteira
        frontier_hover_alloc = [format_alloc_hover(w, active_assets) for w in frontier_weights]

        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=frontier_vols, y=frontier_rets, mode="lines", name="Fronteira Eficiente",
            line=dict(color=CORP["primary"], width=3),
            customdata=frontier_hover_alloc,
            hovertemplate=(
                "<b>Fronteira Eficiente</b><br>"
                "Retorno: %{y:.2%}   |   Volatilidade: %{x:.2%}<br>"
                "<br><b>Alocação nesse ponto:</b><br>%{customdata}"
                "<extra></extra>"
            ),
        ))
        cml_x = [0, vol_maxsh * 1.6]
        cml_y = [rf_rate, rf_rate + shrp_maxsh * vol_maxsh * 1.6]
        fig_ef.add_trace(go.Scatter(
            x=cml_x, y=cml_y, mode="lines", name="Capital Market Line",
            line=dict(color=CORP["warning"], width=1.8, dash="dash"), hoverinfo="skip",
        ))
        fig_ef.add_trace(go.Scatter(
            x=indiv_vols, y=indiv_rets, mode="markers+text",
            name="Ações Individuais", text=active_assets, textposition="top center",
            textfont=dict(size=9, color=CORP["neutral"]),
            marker=dict(size=8, color="#CBD5E1", line=dict(color=CORP["neutral"], width=1)),
            hovertemplate="<b>%{text}</b><br>Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
        ))
        for name_pt, ww, col_pt, sym, sz, text_pt in [
            (f"Atual (Sharpe {shrp_cur:.2f})",       w_cur,    CORP["warning"], "star",    18, "Atual"),
            (f"Máx. Sharpe ({shrp_maxsh:.2f})",       w_maxsh,  CORP["success"], "star",    18, "Máx. Sharpe"),
            (f"Mín. Vol. (Sharpe {shrp_minvol:.2f})", w_minvol, CORP["purple"],  "diamond", 16, "Mín. Vol."),
        ]:
            rv, vv = port_return(ww), port_vol(ww)
            fig_ef.add_trace(go.Scatter(
                x=[vv], y=[rv], mode="markers+text",
                name=name_pt, text=[text_pt], textposition="top right",
                marker=dict(size=sz, color=col_pt, symbol=sym, line=dict(color="white", width=1.5)),
                hovertemplate=f"<b>{text_pt}</b><br>Ret: %{{y:.2%}}<br>Vol: %{{x:.2%}}<extra></extra>",
            ))
        fig_ef.add_trace(go.Scatter(
            x=[0], y=[rf_rate], mode="markers+text", name=f"CDI ({rf_rate:.2%} a.a.)",
            text=["CDI"], textposition="bottom right",
            marker=dict(size=10, color=CORP["danger"], symbol="circle", line=dict(color="white", width=1)),
        ))
        fig_ef.update_layout(**corp_layout(
            title=dict(text="Fronteira Eficiente de Markowitz — Universo de Ações", font_size=14),
            xaxis=dict(title="Volatilidade Anualizada (%)", tickformat=".1%", rangemode="tozero", showgrid=False),
            yaxis=dict(title="Retorno Esperado Anualizado (%)", tickformat=".1%", gridcolor="#F1F5F9"),
            legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font_size=10),
            hovermode="closest",
            hoverlabel=dict(bgcolor="white", bordercolor=CORP["border"], font_size=11,
                             font_color=CORP["text_main"], align="left"),
            height=560, margin=dict(t=60, b=140),
        ))
        st.plotly_chart(fig_ef, use_container_width=True)
        st.caption("💡 Passe o mouse sobre a linha da **Fronteira Eficiente** para ver a alocação "
                    "(peso por ação) da carteira ótima correspondente àquele ponto de risco/retorno.")

        # ── Comparativo de Alocação ───────────────────────────────────────────
        st.markdown("<div class='section-header'>Comparativo de Alocação</div>", unsafe_allow_html=True)
        df_alloc = pd.DataFrame({
            "⭐ Atual":        np.round(w_cur    * 100, 2),
            "🟢 Máx. Sharpe": np.round(w_maxsh  * 100, 2),
            "🟣 Mín. Vol.":   np.round(w_minvol * 100, 2),
        }, index=active_assets)
        df_alloc.index.name = "Ação"

        df_metrics = pd.DataFrame({
            "Retorno (a.a.)":   [f"{ret_cur:.2%}",    f"{ret_maxsh:.2%}",    f"{ret_minvol:.2%}"],
            "Volatilidade":     [f"{vol_cur:.2%}",    f"{vol_maxsh:.2%}",    f"{vol_minvol:.2%}"],
            "Índice de Sharpe": [f"{shrp_cur:.2f}",   f"{shrp_maxsh:.2f}",   f"{shrp_minvol:.2f}"],
        }, index=["⭐ Atual", "🟢 Máx. Sharpe", "🟣 Mín. Vol."])

        col_ef1, col_ef2 = st.columns([3, 2])
        with col_ef1:
            st.markdown("**Alocação por Ação (%)**")
            st.dataframe(
                df_alloc.style.format("{:.1f}%").background_gradient(cmap="Blues", axis=None, vmin=0, vmax=100),
                use_container_width=True, height=min(400, 50 + 35 * n_assets),
            )
        with col_ef2:
            st.markdown("**Métricas Resumidas**")
            st.dataframe(
                df_metrics.style.set_properties(**{"text-align": "center"})
                  .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
                use_container_width=True,
            )

        delta_ret = ret_maxsh - ret_cur
        delta_vol = vol_maxsh - vol_cur
        st.info(
            f"**Potencial de Melhoria → Máx. Sharpe** | "
            f"Retorno: {'▲' if delta_ret >= 0 else '▼'} {abs(delta_ret):.2%} a.a.  |  "
            f"Volatilidade: {'▲' if delta_vol >= 0 else '▼'} {abs(delta_vol):.2%} a.a."
        )

        # ── Walk-Forward ──────────────────────────────────────────────────────
        st.markdown("""
        <hr style='border:none;border-top:1px solid #E2E8F0;margin:20px 0'>
        <div class='section-header'>🔄 Otimização Walk-Forward — Carteira Cenários</div>
        """, unsafe_allow_html=True)
        st.caption("Reotimiza a carteira a cada **6 meses** usando somente dados históricos disponíveis. "
                   "A **Carteira Cenários** é a mediana dos pesos ótimos de cada semestre.")

        with st.spinner("⚙️ Executando Walk-Forward Optimization semestral…"):
            wf_weights_list, wf_window_info = run_walkforward_optimization(
                returns_ef, cdi_ret_series.mean(), window_months=6,
            )

        if len(wf_weights_list) < 2:
            st.warning(f"⚠️ Dados insuficientes para Walk-Forward ({len(wf_weights_list)} janela(s) — mínimo: 2). "
                       "Amplie o período no sidebar para pelo menos 12 meses.")
        else:
            w_cenarios, df_wf_windows = build_scenario_portfolio(wf_weights_list, active_assets, method="median")
            shrp_cen,    vol_cen,    ret_cen    = compute_scenario_metrics(returns_ef, w_cenarios, cdi_ret_series)
            shrp_cur_wf, vol_cur_wf, ret_cur_wf = compute_scenario_metrics(returns_ef, w_cur,      cdi_ret_series)

            df_comp = pd.DataFrame({
                "⭐ Atual":          np.round(w_cur      * 100, 1),
                "🟢 Máx. Sharpe":   np.round(w_maxsh    * 100, 1),
                "🟣 Mín. Vol.":     np.round(w_minvol   * 100, 1),
                "🔵 Cenários (WF)": np.round(w_cenarios * 100, 1),
            }, index=active_assets)
            df_comp.index.name = "Ação"

            st.dataframe(
                df_comp.style.format("{:.1f}%").background_gradient(cmap="Blues", axis=None, vmin=0, vmax=100),
                use_container_width=True, height=min(500, 60 + 35 * n_assets),
            )

            _bar_colors = {
                "⭐ Atual":          CORP["warning"],
                "🟢 Máx. Sharpe":   CORP["success"],
                "🟣 Mín. Vol.":     CORP["purple"],
                "🔵 Cenários (WF)": CORP["primary"],
            }
            fig_wf = go.Figure()
            for col_lbl, color in _bar_colors.items():
                fig_wf.add_trace(go.Bar(
                    name=col_lbl, x=active_assets, y=df_comp[col_lbl],
                    marker_color=color, opacity=0.85,
                    hovertemplate=f"<b>{col_lbl}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
                ))
            fig_wf.update_layout(**corp_layout(
                barmode="group",
                title=dict(text=f"Comparativo de Alocação — {len(wf_weights_list)} janelas semestrais", font_size=13),
                xaxis=dict(title="Ação", tickangle=-30, showgrid=False),
                yaxis=dict(title="Peso (%)", ticksuffix="%", gridcolor="#F1F5F9"),
                legend=dict(orientation="h", y=1.10, x=0.5, xanchor="center", font_size=11),
                height=440, margin=dict(t=90, b=80),
            ))
            st.plotly_chart(fig_wf, use_container_width=True)

            delta_sh = shrp_cen - shrp_cur_wf
            delta_rt = ret_cen  - ret_cur_wf
            delta_vl = vol_cen  - vol_cur_wf
            df_valid = pd.DataFrame({
                "Retorno (a.a.)":      [f"{ret_cur_wf:.2%}", f"{ret_cen:.2%}"],
                "Volatilidade (a.a.)": [f"{vol_cur_wf:.2%}", f"{vol_cen:.2%}"],
                "Índice de Sharpe":    [f"{shrp_cur_wf:.2f}", f"{shrp_cen:.2f}"],
            }, index=["⭐ Carteira Atual", "🔵 Cenários (WF)"])

            col_v1, col_v2 = st.columns([5, 4])
            with col_v1:
                st.dataframe(
                    df_valid.style.set_properties(**{"text-align": "center"})
                      .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
                    use_container_width=True,
                )
            with col_v2:
                _ic = lambda v: "▲" if v >= 0 else "▼"
                st.info(
                    f"**Ganho Cenários vs. Atual**\n\n"
                    f"Sharpe: {_ic(delta_sh)} {abs(delta_sh):.2f}\n\n"
                    f"Retorno: {_ic(delta_rt)} {abs(delta_rt):.2%} a.a.\n\n"
                    f"Volatilidade: {_ic(delta_vl)} {abs(delta_vl):.2%} a.a."
                )

            with st.expander(f"🔍 Detalhe por semestre ({len(wf_weights_list)} janelas — método: mediana)"):
                if wf_window_info:
                    df_wf_disp = df_wf_windows.copy()
                    df_wf_disp.index = [
                        f"S{i+1}: {s.strftime('%b/%Y')} → {e.strftime('%b/%Y')}"
                        for i, (s, e) in enumerate(wf_window_info)
                    ]
                    df_wf_disp.index.name = "Semestre"
                    st.dataframe(
                        (df_wf_disp * 100).style.format("{:.1f}%")
                          .background_gradient(cmap="Blues", axis=None, vmin=0, vmax=100),
                        use_container_width=True,
                    )
                st.caption("💡 A **Carteira Cenários** é a mediana coluna-a-coluna dos pesos acima, normalizada para 100%.")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<hr style='border:none;border-top:1px solid #E2E8F0;margin:32px 0 12px'>
<div style='text-align:center;color:#94A3B8;font-size:11px'>
  Equity Allocator Pro · Dados: yFinance, BCB (SGS 4391) ·
  Resultados históricos não garantem retornos futuros. Não é recomendação de investimento.
</div>
""", unsafe_allow_html=True)
