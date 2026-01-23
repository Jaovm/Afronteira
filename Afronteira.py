import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Configuração da Página
st.set_page_config(page_title="Otimizador de Markowitz", layout="wide")

# ==========================================
# Funções Auxiliares de Cálculo
# ==========================================

def get_data(tickers, period):
    """
    Baixa dados do Yahoo Finance e trata sufixos .SA.
    """
    if not tickers:
        return None
    
    # Tratamento dos tickers (garantir .SA para ações brasileiras se não houver)
    ticker_list = [t.strip().upper() for t in tickers.split(',')]
    processed_tickers = []
    for t in ticker_list:
        if not t.endswith('.SA') and not t.endswith('.JO') and '^' not in t and len(t) < 6:
            # Assunção simples: se for curto e sem sufixo, tenta adicionar .SA
            # (Pode ser ajustado conforme necessidade)
            processed_tickers.append(f"{t}.SA")
        else:
            processed_tickers.append(t)
            
    try:
        data = yf.download(processed_tickers, period=f"{period}y")['Adj Close']
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return None

    # Verifica se algum dado foi baixado
    if data is None or data.empty:
        return None
    
    # Remove colunas vazias (tickers inválidos)
    data = data.dropna(axis=1, how='all')
    
    # Remove linhas com NaN (datas sem negociação para algum ativo)
    data = data.dropna()
    
    return data

def calculate_metrics(data):
    """
    Calcula retornos logarítmicos, média e matriz de covariância.
    """
    # Retornos Logarítmicos
    log_returns = np.log(data / data.shift(1)).dropna()
    
    # Retorno médio anualizado (252 dias úteis)
    mean_returns = log_returns.mean() * 252
    
    # Matriz de covariância anualizada
    cov_matrix = log_returns.cov() * 252
    
    return log_returns, mean_returns, cov_matrix

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Calcula retorno, volatilidade e Sharpe Ratio de uma carteira.
    """
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (returns - risk_free_rate) / std
    return returns, std, sharpe

# ==========================================
# Funções de Otimização (SciPy)
# ==========================================

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Função objetivo para Maximizar Sharpe (Minimizar Negativo)"""
    p_ret, p_var, p_sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return -p_sharpe

def portfolio_volatility(weights, mean_returns, cov_matrix, risk_free_rate):
    """Função objetivo para Minimizar Volatilidade"""
    p_ret, p_var, p_sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return p_var

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, objective_function):
    """
    Executa a otimização via SLSQP.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    
    # Restrições: Soma dos pesos = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Limites: 0 <= peso <= 1 (Sem alavancagem/venda a descoberto)
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    
    # Chute inicial (pesos iguais)
    init_guess = num_assets * [1. / num_assets,]
    
    result = minimize(objective_function, init_guess, args=args, 
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result

# ==========================================
# Interface do Usuário
# ==========================================

st.title("📈 Otimizador de Carteira (Markowitz)")
st.markdown("---")

# Sidebar - Inputs
st.sidebar.header("Parâmetros da Carteira")

input_tickers = st.sidebar.text_area(
    "Insira os Tickers (separados por vírgula)", 
    value="PETR4, VALE3, WEGE3, ITUB4, ABEV3"
)

input_period = st.sidebar.slider("Histórico (Anos)", min_value=1, max_value=10, value=2)
input_rf = st.sidebar.number_input("Taxa Livre de Risco Anual (%)", value=10.75, step=0.25)
rf_rate = input_rf / 100

btn_process = st.sidebar.button("Otimizar Carteira")

if btn_process:
    with st.spinner('Baixando dados e processando...'):
        # 1. Obtenção de Dados
        df_prices = get_data(input_tickers, input_period)
        
        if df_prices is None or len(df_prices.columns) < 2:
            st.error("Erro: Dados insuficientes. Insira pelo menos 2 tickers válidos.")
        else:
            # 2. Cálculos Estatísticos
            log_ret, mu, S = calculate_metrics(df_prices)
            tickers_found = df_prices.columns.tolist()
            
            st.success(f"Dados baixados com sucesso para {len(tickers_found)} ativos: {', '.join(tickers_found)}")
            
            # 3. Otimizações
            
            # A) Máximo Sharpe Ratio
            max_sharpe_result = optimize_portfolio(mu, S, rf_rate, neg_sharpe_ratio)
            w_sharpe = max_sharpe_result.x
            ret_sharpe, vol_sharpe, sr_sharpe = portfolio_performance(w_sharpe, mu, S, rf_rate)
            
            # B) Mínima Variância
            min_vol_result = optimize_portfolio(mu, S, rf_rate, portfolio_volatility)
            w_min_vol = min_vol_result.x
            ret_min_vol, vol_min_vol, sr_min_vol = portfolio_performance(w_min_vol, mu, S, rf_rate)
            
            # C) Maior Retorno Esperado (Corner Solution em Long-Only)
            # Em Markowitz sem limites superiores < 100%, o max retorno é 100% no ativo de maior média
            idx_max_ret = np.argmax(mu)
            w_max_ret = np.zeros(len(mu))
            w_max_ret[idx_max_ret] = 1.0
            ret_max_ret, vol_max_ret, sr_max_ret = portfolio_performance(w_max_ret, mu, S, rf_rate)

            # ==========================================
            # Visualização dos Resultados
            # ==========================================
            
            st.markdown("### 📊 Comparativo de Carteiras Ótimas")
            
            # DataFrame de Métricas
            metrics_data = {
                "Métrica": ["Retorno Esperado (a.a.)", "Volatilidade (a.a.)", "Sharpe Ratio"],
                "Máximo Sharpe": [f"{ret_sharpe:.2%}", f"{vol_sharpe:.2%}", f"{sr_sharpe:.2f}"],
                "Mínima Volatilidade": [f"{ret_min_vol:.2%}", f"{vol_min_vol:.2%}", f"{sr_min_vol:.2f}"],
                "Máximo Retorno": [f"{ret_max_ret:.2%}", f"{vol_max_ret:.2%}", f"{sr_max_ret:.2f}"]
            }
            st.table(pd.DataFrame(metrics_data).set_index("Métrica"))

            st.markdown("### ⚖️ Alocação de Ativos (Pesos)")
            
            # DataFrame de Pesos
            weights_df = pd.DataFrame({
                "Ativo": tickers_found,
                "Max Sharpe": w_sharpe,
                "Min Volatilidade": w_min_vol,
                "Max Retorno": w_max_ret
            }).set_index("Ativo")
            
            # Formatação e Exibição
            st.dataframe(weights_df.style.format("{:.2%}")
                         .background_gradient(cmap='Greens', axis=0))

            # ==========================================
            # Gráficos
            # ==========================================
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔥 Correlação entre Ativos")
                fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                sns.heatmap(log_ret.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_corr)
                st.pyplot(fig_corr)
                
            with col2:
                st.markdown("#### 🚀 Fronteira Eficiente")
                
                # Simulação de Monte Carlo para plotar a nuvem de pontos
                num_portfolios = 5000
                all_weights = np.zeros((num_portfolios, len(mu)))
                ret_arr = np.zeros(num_portfolios)
                vol_arr = np.zeros(num_portfolios)
                sharpe_arr = np.zeros(num_portfolios)

                for i in range(num_portfolios):
                    # Pesos aleatórios
                    weights = np.array(np.random.random(len(mu)))
                    weights = weights / np.sum(weights)
                    all_weights[i,:] = weights
                    
                    # Métricas
                    ret_arr[i], vol_arr[i], sharpe_arr[i] = portfolio_performance(weights, mu, S, rf_rate)

                fig_ef, ax_ef = plt.subplots(figsize=(8, 6))
                
                # Scatter plot da nuvem
                sc = ax_ef.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', marker='.', alpha=0.3)
                plt.colorbar(sc, label='Sharpe Ratio')
                
                # Plotar pontos ótimos
                ax_ef.scatter(vol_sharpe, ret_sharpe, marker='*', color='r', s=200, label='Max Sharpe')
                ax_ef.scatter(vol_min_vol, ret_min_vol, marker='*', color='b', s=200, label='Min Volatilidade')
                ax_ef.scatter(vol_max_ret, ret_max_ret, marker='*', color='g', s=200, label='Max Retorno')
                
                ax_ef.set_title(f'Fronteira Eficiente (Simulação {num_portfolios} Carteiras)')
                ax_ef.set_xlabel('Volatilidade Anual')
                ax_ef.set_ylabel('Retorno Esperado Anual')
                ax_ef.legend(loc='best')
                ax_ef.grid(True, linestyle='--', alpha=0.6)
                
                st.pyplot(fig_ef)

else:
    st.info("Utilize a barra lateral para configurar os parâmetros e clique em 'Otimizar Carteira'.")
    
