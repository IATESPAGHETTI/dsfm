

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np
from arch import arch_model
from sklearn.cluster import KMeans
from stocknews import StockNews

st.set_page_config(layout="wide")

def get_stock_data(tickers, start_date, end_date):
    """Fetches historical stock data from Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

def plot_pie_chart(df):
    """Plots a pie chart of the portfolio allocation."""
    fig = px.pie(df, values='Current Value', names='Ticker', title='Portfolio Allocation')
    return fig

def garch_volatility_forecast(returns):
    """Forecasts volatility using a GARCH(1,1) model."""
    model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    model_fit = model.fit(disp='off')
    forecast = model_fit.forecast(horizon=30)
    return np.sqrt(forecast.variance.iloc[-1].mean()) * 100, model_fit

def kmeans_regime_analysis(data):
    """Performs K-Means clustering to identify market regimes."""
    if isinstance(data.columns, pd.MultiIndex):
        close_column = [col for col in data.columns if col[0] == 'Close'][0]
    else:
        close_column = 'Close'
    returns = data[close_column].pct_change()
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(returns.dropna().values.reshape(-1, 1))
    data['Regime'] = np.nan
    data.loc[returns.dropna().index, 'Regime'] = kmeans.labels_
    data['Regime'] = data['Regime'].ffill()
    return data

def plot_regime_chart(df):
    """Plots the portfolio performance with market regimes."""
    if isinstance(df.columns, pd.MultiIndex):
        close_column_name = [col for col in df.columns if col[0] == 'Close'][0]
        y_data = df[close_column_name]
    else:
        y_data = df['Close']
    fig = px.line(x=df.index, y=y_data, color=df['Regime'], title='Portfolio Performance with Market Regimes')
    fig.update_layout(xaxis_title='Date', yaxis_title='Price')
    return fig

st.title('📈 Portfolio Risk & Regime Dashboard')

st.sidebar.header('Upload Portfolio')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    portfolio_df = pd.read_csv(uploaded_file)

    if 'Ticker' not in portfolio_df.columns or 'Quantity' not in portfolio_df.columns:
        st.error("CSV file must have 'Ticker' and 'Quantity' columns.")
    else:
        tickers = portfolio_df['Ticker'].tolist()
        end_date = pd.to_datetime('today')
        start_date = end_date - pd.DateOffset(years=10)

        with st.spinner('Fetching data...'):
            stock_data = get_stock_data(tickers, start_date, end_date)

        if stock_data.empty:
            st.error("Could not download data for the given tickers. Please check the ticker symbols.")
        else:
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📉 Volatility Forecast", "📈 Regime Analysis", "📰 News Analysis"])

            with tab1:
                st.header('Portfolio Overview')
                last_prices = stock_data['Close'].iloc[-1]
                portfolio_df['Current Price'] = portfolio_df['Ticker'].map(last_prices)
                portfolio_df['Current Value'] = portfolio_df['Current Price'] * portfolio_df['Quantity']
                total_portfolio_value = portfolio_df['Current Value'].sum()

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}")
                    st.plotly_chart(plot_pie_chart(portfolio_df), use_container_width=True)
                
                with col2:
                    st.dataframe(portfolio_df, use_container_width=True)

            with tab2:
                st.header('Volatility Forecast (30-Day)')
                st.write("This section uses a GARCH(1,1) model to forecast the 30-day volatility of each stock in your portfolio. GARCH (Generalized Autoregressive Conditional Heteroskedasticity) is a statistical model used to analyze time series data where the variance error is believed to be serially autocorrelated.")
                volatility_data = []
                for ticker in tickers:
                    returns = stock_data['Close'][ticker].pct_change().dropna() * 100
                    forecasted_volatility, model_fit = garch_volatility_forecast(returns)
                    current_holding = portfolio_df[portfolio_df['Ticker'] == ticker]['Current Value'].iloc[0]
                    volatility_data.append({
                        'Ticker': ticker,
                        'Current Holding': f"${current_holding:,.2f}",
                        '30-Day Forecasted Volatility': f"{forecasted_volatility:.2f}%"
                    })
                    with st.expander(f"GARCH Model Summary for {ticker}"):
                        st.text(str(model_fit.summary()))

                st.table(pd.DataFrame(volatility_data))

            with tab3:
                st.header('Market Regime Analysis')
                st.write("This section uses K-Means clustering to identify different market regimes (e.g., 'Calm' and 'Stressed') based on the historical returns of a market index (S&P 500). This helps you understand how your portfolio might perform under different market conditions.")
                with st.spinner('Running regime analysis...'):
                    market_index_data = get_stock_data('^GSPC', start_date, end_date)
                if not market_index_data.empty:
                    regime_data = kmeans_regime_analysis(market_index_data)
                    st.plotly_chart(plot_regime_chart(regime_data), use_container_width=True)

                    regime_returns = regime_data.groupby('Regime')['Close'].pct_change().dropna()
                    avg_returns = regime_returns.groupby(regime_data['Regime']).mean()

                    st.write("Average Daily Returns per Regime:")
                    st.table(avg_returns)

                    with st.expander("View Regime Data"):
                        st.dataframe(regime_data, use_container_width=True)
                else:
                    st.warning("Could not download market index data for regime analysis.")

            with tab4:
                st.header('News Analysis')
                for ticker in tickers:
                    st.subheader(f'News for {ticker}')
                    try:
                        sn = StockNews(ticker, save_news=False)
                        df_news = sn.read_rss()
                        for i in range(min(5, len(df_news))):
                            with st.expander(f"{df_news['title'][i]}"):
                                st.write(df_news['published'][i])
                                st.write(df_news['summary'][i])
                    except Exception as e:
                        st.error(f"Could not fetch news for {ticker}: {e}")
