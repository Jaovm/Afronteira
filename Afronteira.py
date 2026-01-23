import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Função para baixar dados históricos
def download_data(tickers, period_years):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=period_years)
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Função para calcular retornos logarítmicos
def calculate_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

# Função para anualizar retornos e volatilidade
def annualize_returns(returns, num_periods=252):
    return returns.mean() * num_periods

def annualize_volatility(returns, num_periods=252):
    return returns.std() * np.sqrt(num_periods)

# Função para calcular retorno da carteira
def portfolio_return(weights, returns):
    return np.dot(weights, returns)

# Função para calcular volatilidade da carteira
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Função para calcular Sharpe Ratio
def sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    ret = portfolio_return(weights, returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return (ret - risk_free_rate) / vol

# Função de otimização genérica
def optimize_portfolio(objective_func, returns, cov_matrix, risk_free_rate, bounds, constraints):
    num_assets = len(returns)
    initial_guess = np.array(num_assets * [1. / num_assets])
    result = minimize(objective_func, initial_guess, args=(returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Objetivos de otimização
def neg_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, returns, cov_matrix, risk_free_rate)

def neg_portfolio_return(weights, returns, cov_matrix, risk_free_rate):
    return -portfolio_return(weights, returns)

def portfolio_variance(weights, returns, cov_matrix, risk_free_rate):
    return portfolio_volatility(weights, cov_matrix) ** 2

# Função para gerar fronteira eficiente
def efficient_frontier(returns, cov_matrix, num_portfolios=100):
    target_returns = np.linspace(returns.min(), returns.max(), num_portfolios)
    efficient_portfolios = []
    bounds = [(0, 1) for _ in returns]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    for target in target_returns:
        cons = constraints.copy()
        cons = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, returns) - target},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        result = minimize(portfolio_variance, np.array(len(returns) * [1. / len(returns)]),
                          args=(returns, cov_matrix, 0), method='SLSQP', bounds=bounds, constraints=cons)
        if result.success:
            efficient_portfolios.append((target, np.sqrt(result.fun)))
    return efficient_portfolios

# Configuração da interface Streamlit
st.title("Otimização de Carteira de Investimentos - Modelo de Markowitz")

# Inputs do usuário
tickers_input = st.text_input("Tickers da carteira (separados por vírgulas, ex: PETR4, VALE3, ITUB4)", "")
period_years = st.number_input("Período histórico de análise (em anos)", min_value=1, value=5)
risk_free_rate = st.number_input("Taxa livre de risco (em decimal, ex: 0.05 para 5%)", value=0.05)

if st.button("Otimizar Carteira"):
    if not tickers_input:
        st.error("Por favor, insira os tickers.")
    else:
        try:
            # Processar tickers
            tickers = [t.strip().upper() + ('.SA' if not t.strip().upper().endswith('.SA') else '') for t in tickers_input.split(',')]
            
            # Baixar dados
            prices = download_data(tickers, period_years)
            if prices.empty or prices.isnull().all().all():
                raise ValueError("Dados históricos não disponíveis para os tickers informados.")
            
            # Calcular retornos logarítmicos
            log_returns = calculate_log_returns(prices)
            
            # Matriz de covariância e correlação (anualizadas)
            cov_matrix = log_returns.cov() * 252
            corr_matrix = log_returns.corr()
            
            # Retornos esperados anualizados
            expected_returns = annualize_returns(log_returns)
            
            # Volatilidade anualizada (não usada diretamente)
            # volatility = annualize_volatility(log_returns)
            
            # Configurações de otimização
            num_assets = len(tickers)
            bounds = [(0, 1) for _ in range(num_assets)]
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Otimização para maior Sharpe Ratio
            max_sharpe = optimize_portfolio(neg_sharpe_ratio, expected_returns, cov_matrix, risk_free_rate, bounds, constraints)
            
            # Otimização para maior retorno esperado
            max_return = optimize_portfolio(neg_portfolio_return, expected_returns, cov_matrix, risk_free_rate, bounds, constraints)
            
            # Otimização para menor risco (variância mínima)
            min_variance = optimize_portfolio(portfolio_variance, expected_returns, cov_matrix, risk_free_rate, bounds, constraints)
            
            # Função para extrair resultados
            def get_results(opt_result, returns, cov_matrix, risk_free_rate):
                weights = opt_result.x
                ret = portfolio_return(weights, returns)
                vol = portfolio_volatility(weights, cov_matrix)
                sharpe = (ret - risk_free_rate) / vol
                return weights, ret, vol, sharpe
            
            # Resultados
            sharpe_weights, sharpe_ret, sharpe_vol, sharpe_sr = get_results(max_sharpe, expected_returns, cov_matrix, risk_free_rate)
            return_weights, return_ret, return_vol, return_sr = get_results(max_return, expected_returns, cov_matrix, risk_free_rate)
            variance_weights, variance_ret, variance_vol, variance_sr = get_results(min_variance, expected_returns, cov_matrix, risk_free_rate)
            
            # Apresentar resultados em tabelas
            results_df = pd.DataFrame({
                'Carteira': ['Maior Sharpe Ratio', 'Maior Retorno Esperado', 'Menor Risco'],
                'Retorno Anual Esperado': [sharpe_ret, return_ret, variance_ret],
                'Volatilidade Anual': [sharpe_vol, return_vol, variance_vol],
                'Sharpe Ratio': [sharpe_sr, return_sr, variance_sr]
            })
            st.subheader("Resultados das Carteiras Otimizadas")
            st.dataframe(results_df.style.format("{:.4f}"))
            
            # Pesos ótimos
            weights_df = pd.DataFrame({
                'Ativo': tickers,
                'Maior Sharpe Ratio': sharpe_weights,
                'Maior Retorno Esperado': return_weights,
                'Menor Risco': variance_weights
            })
            st.subheader("Pesos Ótimos por Ativo")
            st.dataframe(weights_df.style.format("{:.4f}", subset=weights_df.columns[1:]))
            
            # Visualizações
            st.subheader("Heatmap de Correlação entre Ativos")
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            
            # Fronteira Eficiente (opcional)
            st.subheader("Fronteira Eficiente")
            efficient = efficient_frontier(expected_returns, cov_matrix)
            if efficient:
                rets, vols = zip(*efficient)
                fig_ef, ax_ef = plt.subplots()
                ax_ef.plot(vols, rets, 'b--')
                ax_ef.scatter(sharpe_vol, sharpe_ret, color='green', label='Maior Sharpe')
                ax_ef.scatter(return_vol, return_ret, color='blue', label='Maior Retorno')
                ax_ef.scatter(variance_vol, variance_ret, color='red', label='Menor Risco')
                ax_ef.set_xlabel('Volatilidade')
                ax_ef.set_ylabel('Retorno Esperado')
                ax_ef.legend()
                st.pyplot(fig_ef)
            else:
                st.warning("Não foi possível gerar a fronteira eficiente.")
        
        except Exception as e:
            st.error(f"Erro ao processar os dados: {str(e)}")
