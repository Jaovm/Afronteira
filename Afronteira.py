import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Configuração da página
st.set_page_config(page_title="Otimizador de Carteira Markowitz", layout="wide")

# Estilo para os gráficos
plt.style.use('ggplot')

def get_stock_data(tickers, start_date, end_date):
    """Baixa dados do Yahoo Finance com tratamento para tickers B3."""
    data = pd.DataFrame()
    valid_tickers = []
    
    for t in tickers:
        t = t.strip().upper()
        # Tentativa simples de identificar tickers BR sem sufixo
        if not t.endswith('.SA') and len(t) <= 6 and any(char.isdigit() for char in t):
             # Verifica se é um ticker comum da B3 (ex: PETR4)
            ticker_sa = f"{t}.SA"
        else:
            ticker_sa = t
            
        try:
            # Tenta baixar com .SA (prioridade se parecer BR) ou original
            temp = yf.download(ticker_sa, start=start_date, end=end_date, progress=False)
            if not temp.empty:
                data[t] = temp['Adj Close']
                valid_tickers.append(t)
            else:
                # Fallback: se falhou com .SA, tenta sem (ou vice-versa se necessário, mas simplificado aqui)
                temp = yf.download(t, start=start_date, end=end_date, progress=False)
                if not temp.empty:
                    data[t] = temp['Adj Close']
                    valid_tickers.append(t)
        except Exception:
            continue
            
    return data, valid_tickers

def calculate_metrics(weights, mean_returns, cov_matrix, risk_free_rate):
    """Calcula retorno, volatilidade e Sharpe Ratio da carteira."""
    weights = np.array(weights)
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Função objetivo para maximizar Sharpe (minimizar negativo)."""
    p_ret, p_vol, p_sr = calculate_metrics(weights, mean_returns, cov_matrix, risk_free_rate)
    return -p_sr

def portfolio_volatility_func(weights, mean_returns, cov_matrix, risk_free_rate):
    """Função objetivo para minimizar volatilidade."""
    p_ret, p_vol, p_sr = calculate_metrics(weights, mean_returns, cov_matrix, risk_free_rate)
    return p_vol

def neg_portfolio_return(weights, mean_returns, cov_matrix, risk_free_rate):
    """Função objetivo para maximizar retorno (minimizar negativo)."""
    p_ret, p_vol, p_sr = calculate_metrics(weights, mean_returns, cov_matrix, risk_free_rate)
    return -p_ret

def optimize_portfolio(mean_returns, cov_matrix, num_assets, risk_free_rate, objective_function):
    """Executa a otimização via SciPy."""
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]
    
    result = minimize(objective_function, init_guess, 
                      args=(mean_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# --- Interface Sidebar ---
st.sidebar.header("Parâmetros da Carteira")

input_tickers = st.sidebar.text_area("Tickers (separados por vírgula)", "PETR4, VALE3, WEGE3, ITUB4, BOVA11")
years = st.sidebar.number_input("Período de análise (anos)", min_value=1, max_value=10, value=3)
risk_free = st.sidebar.number_input("Taxa Livre de Risco Anual (%)", min_value=0.0, max_value=20.0, value=10.0) / 100

run_btn = st.sidebar.button("Otimizar Carteira")

# --- Lógica Principal ---
if run_btn:
    ticker_list = [x.strip() for x in input_tickers.split(',') if x.strip()]
    
    if len(ticker_list) < 2:
        st.error("Por favor, insira pelo menos 2 ativos para otimização.")
    else:
        with st.spinner('Baixando dados e calculando...'):
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=years)
            
            df_prices, valid_tickers = get_stock_data(ticker_list, start_date, end_date)
            
            if df_prices.empty or len(valid_tickers) < 2:
                st.error("Não foi possível baixar dados suficientes. Verifique os tickers.")
            else:
                st.success(f"Dados baixados para: {', '.join(valid_tickers)}")
                
                # Cálculos Estatísticos
                log_returns = np.log(df_prices / df_prices.shift(1)).dropna()
                mean_returns = log_returns.mean()
                cov_matrix = log_returns.cov()
                corr_matrix = log_returns.corr()
                num_assets = len(valid_tickers)

                # --- Otimizações ---
                
                # 1. Max Sharpe Ratio
                opt_sharpe = optimize_portfolio(mean_returns, cov_matrix, num_assets, risk_free, neg_sharpe_ratio)
                ret_sharpe, vol_sharpe, sr_sharpe = calculate_metrics(opt_sharpe.x, mean_returns, cov_matrix, risk_free)
                
                # 2. Mínima Variância (Risco)
                opt_min_vol = optimize_portfolio(mean_returns, cov_matrix, num_assets, risk_free, portfolio_volatility_func)
                ret_min_vol, vol_min_vol, sr_min_vol = calculate_metrics(opt_min_vol.x, mean_returns, cov_matrix, risk_free)
                
                # 3. Retorno Máximo
                opt_max_ret = optimize_portfolio(mean_returns, cov_matrix, num_assets, risk_free, neg_portfolio_return)
                ret_max_ret, vol_max_ret, sr_max_ret = calculate_metrics(opt_max_ret.x, mean_returns, cov_matrix, risk_free)

                # --- Exibição dos Resultados ---
                
                st.subheader("Resultados das Carteiras Otimizadas")
                
                col1, col2, col3 = st.columns(3)
                
                # Formatação de dados para exibição
                def format_metrics(ret, vol, sr):
                    return f"Retorno: {ret:.2%}", f"Volatilidade: {vol:.2%}", f"Sharpe: {sr:.2f}"

                r_s, v_s, s_s = format_metrics(ret_sharpe, vol_sharpe, sr_sharpe)
                r_v, v_v, s_v = format_metrics(ret_min_vol, vol_min_vol, sr_min_vol)
                r_r, v_r, s_r = format_metrics(ret_max_ret, vol_max_ret, sr_max_ret)

                with col1:
                    st.info(f"**Máximo Sharpe**\n\n{r_s}\n\n{v_s}\n\n{s_s}")
                with col2:
                    st.warning(f"**Mínima Volatilidade**\n\n{r_v}\n\n{v_v}\n\n{s_v}")
                with col3:
                    st.success(f"**Máximo Retorno**\n\n{r_r}\n\n{v_r}\n\n{s_r}")

                # Tabela de Pesos
                st.subheader("Alocação de Ativos (Pesos %)")
                weights_df = pd.DataFrame({
                    "Max Sharpe": opt_sharpe.x * 100,
                    "Min Volatilidade": opt_min_vol.x * 100,
                    "Max Retorno": opt_max_ret.x * 100
                }, index=valid_tickers)
                st.dataframe(weights_df.style.format("{:.2f}%"))

                # --- Visualizações ---
                
                col_graph1, col_graph2 = st.columns(2)
                
                # 1. Heatmap de Correlação
                with col_graph1:
                    st.markdown("### Correlação entre Ativos")
                    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                    cax = ax_corr.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                    fig_corr.colorbar(cax)
                    
                    ticks = np.arange(0, len(valid_tickers), 1)
                    ax_corr.set_xticks(ticks)
                    ax_corr.set_yticks(ticks)
                    ax_corr.set_xticklabels(valid_tickers, rotation=45, ha="left")
                    ax_corr.set_yticklabels(valid_tickers)
                    
                    # Adicionar valores no heatmap
                    for i in range(len(valid_tickers)):
                        for j in range(len(valid_tickers)):
                            text = ax_corr.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                                           ha="center", va="center", color="black", fontsize=8)
                            
                    st.pyplot(fig_corr)

                # 2. Fronteira Eficiente (Monte Carlo)
                with col_graph2:
                    st.markdown("### Fronteira Eficiente")
                    
                    # Simulação Monte Carlo para fundo do gráfico
                    num_simulations = 5000
                    all_weights = np.zeros((num_simulations, num_assets))
                    ret_arr = np.zeros(num_simulations)
                    vol_arr = np.zeros(num_simulations)
                    sharpe_arr = np.zeros(num_simulations)

                    for i in range(num_simulations):
                        w = np.random.random(num_assets)
                        w /= np.sum(w)
                        all_weights[i,:] = w
                        r, v, s = calculate_metrics(w, mean_returns, cov_matrix, risk_free)
                        ret_arr[i] = r
                        vol_arr[i] = v
                        sharpe_arr[i] = s

                    fig_ef, ax_ef = plt.subplots(figsize=(8, 6))
                    sc = ax_ef.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', s=10, alpha=0.5, label='Carteiras Aleatórias')
                    plt.colorbar(sc, label='Sharpe Ratio')
                    
                    # Plotar pontos ótimos
                    ax_ef.scatter(vol_sharpe, ret_sharpe, c='red', s=100, marker='*', label='Max Sharpe')
                    ax_ef.scatter(vol_min_vol, ret_min_vol, c='blue', s=100, marker='D', label='Min Volatilidade')
                    ax_ef.scatter(vol_max_ret, ret_max_ret, c='green', s=100, marker='^', label='Max Retorno')
                    
                    ax_ef.set_title('Fronteira Eficiente (Simulação)')
                    ax_ef.set_xlabel('Volatilidade Anual')
                    ax_ef.set_ylabel('Retorno Esperado Anual')
                    ax_ef.legend()
                    
                    st.pyplot(fig_ef)

else:
    st.info("Utilize a barra lateral para configurar e gerar a carteira.")
