import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as optimize
import plotly.graph_objects as go

# ==============================================================================
# FUNÇÕES DE CÁLCULO E OTIMIZAÇÃO
# ==============================================================================

def get_data(tickers, start_date, end_date):
    """Baixa os dados ajustados do Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def min_func_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    """Função Objetivo: Maximizar Sharpe (Minimizar Sharpe Negativo)."""
    p_ret = np.sum(weights * mean_returns) * 252
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = (p_ret - risk_free_rate) / p_vol
    return -sharpe

def min_func_sharpe_distributed(weights, mean_returns, cov_matrix, risk_free_rate, lambda_reg=2.0):
    """
    Função Objetivo: Maximizar Sharpe com Regularização L2 (Ridge).
    
    Objetivo: Minimizar (-Sharpe + lambda * sum(weights^2))
    
    Racional:
    A maximização tradicional do Sharpe tende a concentrar pesos nos ativos de melhor 
    performance histórica (soluções de canto). Ao adicionar a penalidade (lambda * sum(w^2)),
    criamos uma 'pressão' matemática que favorece a diversificação (pesos distribuídos),
    impedindo que ativos fiquem 'presos' apenas no limite mínimo definido.
    """
    # 1. Métricas da Carteira
    p_ret = np.sum(weights * mean_returns) * 252
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    # 2. Sharpe Ratio
    sharpe = (p_ret - risk_free_rate) / p_vol
    
    # 3. Penalidade por Concentração (L2 Regularization)
    # Quanto mais concentrada a carteira, maior a soma dos quadrados.
    penalty = lambda_reg * np.sum(weights**2)
    
    # Retorna valor a minimizar
    return -sharpe + penalty

def min_func_volatility(weights, mean_returns, cov_matrix, risk_free_rate):
    """Função Objetivo: Minimizar Volatilidade."""
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return p_vol

def min_func_return(weights, mean_returns, cov_matrix, risk_free_rate):
    """Função Objetivo: Maximizar Retorno (Minimizar Retorno Negativo)."""
    p_ret = np.sum(weights * mean_returns) * 252
    return -p_ret

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Otimizador de Carteira", layout="wide")

st.title("📈 Otimizador de Carteira - Markowitz & Diversificação")
st.markdown("""
Esta ferramenta utiliza a Teoria Moderna do Portfólio para encontrar a alocação ótima de ativos.
Agora inclui algoritmos de regularização para evitar concentração excessiva.
""")

# --- Sidebar: Parâmetros de Entrada ---
st.sidebar.header('Parâmetros do Portfólio')

# Input dos Tickers
tickers_input = st.sidebar.text_area(
    'Insira os Tickers (separados por vírgula):', 
    'PETR4.SA, VALE3.SA, WEGE3.SA, ITUB4.SA, BBDC4.SA, ABEV3.SA'
)
tickers = [t.strip() for t in tickers_input.split(',')]

# Datas
start_date = st.sidebar.date_input('Data de Início', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('Data Final', pd.to_datetime('today'))

# Taxa Livre de Risco
risk_free_rate = st.sidebar.number_input('Taxa Livre de Risco Anual (decimal)', value=0.10, step=0.01)

# Restrições de Peso
st.sidebar.subheader('Restrições de Alocação')
min_wt = st.sidebar.slider('Peso Mínimo por Ativo', 0.0, 0.5, 0.01, step=0.01)
max_wt = st.sidebar.slider('Peso Máximo por Ativo', 0.0, 1.0, 1.0, step=0.05)

# Objetivo da Otimização
goal = st.sidebar.selectbox(
    'Objetivo da Otimização',
    (
        'Maximizar Sharpe Ratio', 
        'Maximizar Sharpe Ratio (todos os ativos)', # NOVA OPÇÃO
        'Minimizar Volatilidade', 
        'Maximizar Retorno'
    )
)

# Botão de Execução
run_button = st.sidebar.button('Otimizar Carteira')

# ==============================================================================
# LÓGICA PRINCIPAL
# ==============================================================================

if run_button:
    with st.spinner('Baixando dados e otimizando...'):
        try:
            # 1. Obtenção e Tratamento dos Dados
            df = get_data(tickers, start_date, end_date)
            
            if df.empty:
                st.error("Não foi possível baixar dados. Verifique os tickers.")
                st.stop()
            
            # Cálculo de retornos diários e matriz de covariância
            returns = df.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            num_assets = len(tickers)

            # 2. Configuração da Otimização (Scipy)
            
            # Chute inicial (Equal Weights)
            init_guess = num_assets * [1. / num_assets,]
            
            # Bounds (Limites min e max para cada ativo)
            bounds = tuple((min_wt, max_wt) for asset in range(num_assets))
            
            # Constraints (Soma dos pesos deve ser igual a 1)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

            # Seleção da Função Objetivo
            if goal == 'Maximizar Sharpe Ratio':
                fun_to_optimize = min_func_sharpe
                args_optimization = (mean_returns, cov_matrix, risk_free_rate)
                
            elif goal == 'Maximizar Sharpe Ratio (todos os ativos)':
                # Aqui usamos a função com regularização
                fun_to_optimize = min_func_sharpe_distributed
                # O parâmetro lambda_reg=2.0 está fixo dentro da função ou pode ser passado aqui se alterarmos a assinatura
                args_optimization = (mean_returns, cov_matrix, risk_free_rate)
                
            elif goal == 'Minimizar Volatilidade':
                fun_to_optimize = min_func_volatility
                args_optimization = (mean_returns, cov_matrix, risk_free_rate)
                
            elif goal == 'Maximizar Retorno':
                fun_to_optimize = min_func_return
                args_optimization = (mean_returns, cov_matrix, risk_free_rate)

            # 3. Execução da Otimização
            result = optimize.minimize(
                fun_to_optimize,
                init_guess,
                args=args_optimization,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if not result.success:
                st.warning(f"A otimização pode não ter convergido perfeitamente: {result.message}")

            opt_weights = result.x

            # 4. Cálculo das Métricas Finais da Carteira Otimizada
            opt_ret = np.sum(opt_weights * mean_returns) * 252
            opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights))) * np.sqrt(252)
            opt_sharpe = (opt_ret - risk_free_rate) / opt_vol

            # ==============================================================================
            # EXIBIÇÃO DOS RESULTADOS
            # ==============================================================================
            
            st.success("Otimização Concluída!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Retorno Esperado (a.a.)", f"{opt_ret:.2%}")
            col2.metric("Volatilidade (a.a.)", f"{opt_vol:.2%}")
            col3.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")

            # --- Tabela de Pesos ---
            st.subheader("Alocação Sugerida")
            
            # Criar DataFrame para exibição
            df_weights = pd.DataFrame({
                'Ticker': tickers,
                'Peso': opt_weights
            })
            
            # Ordenar e formatar
            df_weights = df_weights.sort_values(by='Peso', ascending=False)
            df_weights['Peso (%)'] = (df_weights['Peso'] * 100).map('{:.2f}%'.format)
            
            # Exibir lado a lado com gráfico de pizza
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.dataframe(df_weights[['Ticker', 'Peso (%)']], use_container_width=True)
                
            with c2:
                fig_pie = go.Figure(data=[go.Pie(labels=df_weights['Ticker'], values=df_weights['Peso'], hole=.3)])
                fig_pie.update_layout(title_text="Distribuição da Carteira")
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- Backtest Simples (Gráfico de Linha) ---
            st.subheader("Performance Histórica Acumulada (Simulação)")
            
            # Retorno da Carteira Diária
            portfolio_daily_returns = (returns * opt_weights).sum(axis=1)
            cumulative_returns = (1 + portfolio_daily_returns).cumprod()
            
            # Gráfico de Linha
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Carteira Otimizada'))
            fig_line.update_layout(title='Crescimento de R$ 1,00', yaxis_title='Retorno Acumulado', xaxis_title='Data')
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Exibir explicação extra se a nova opção foi escolhida
            if goal == 'Maximizar Sharpe Ratio (todos os ativos)':
                st.info("""
                ℹ️ **Nota Técnica:** Esta otimização utilizou Regularização L2 para penalizar concentrações excessivas. 
                Observe que os pesos tendem a ser distribuídos de forma mais equilibrada entre os ativos, 
                evitando 'soluções de canto' onde a maioria dos ativos ficaria apenas no peso mínimo.
                """)

        except Exception as e:
            st.error(f"Ocorreu um erro durante a execução: {e}")
