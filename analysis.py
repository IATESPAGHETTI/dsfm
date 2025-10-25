
import pandas as pd
import yfinance as yf
import numpy as np
from arch import arch_model
from sklearn.cluster import KMeans
from stocknews import StockNews

def get_stock_data(tickers, start_date, end_date):
    """Fetches historical stock data from Yahoo Finance.""" 
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

def garch_volatility_forecast(returns):
    """Forecasts volatility using a GARCH(1,1) model."""
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    forecast = model_fit.forecast(horizon=30)
    return np.sqrt(forecast.variance.iloc[-1].mean()) * 100

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
    data['Regime'] = data['Regime'].fillna(method='ffill')
    return data

def run_analysis():
    """Runs the portfolio analysis."""
    try:
        portfolio_df = pd.read_csv("C:\\Users\\ASUS\\Desktop\\dsfm_GEM\\sample_portfolio.csv")
    except FileNotFoundError:
        print("Error: sample_portfolio.csv not found. Please make sure the file is in the same directory as the script.")
        return

    if 'Ticker' not in portfolio_df.columns or 'Quantity' not in portfolio_df.columns:
        print("Error: CSV file must have 'Ticker' and 'Quantity' columns.")
        return

    tickers = portfolio_df['Ticker'].tolist()
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.DateOffset(years=10)

    stock_data = get_stock_data(tickers, start_date, end_date)

    if stock_data.empty:
        print("Error: Could not download data for the given tickers. Please check the ticker symbols.")
        return

    print("--- Portfolio Overview ---")
    last_prices = stock_data['Close'].iloc[-1]
    portfolio_df['Current Price'] = portfolio_df['Ticker'].map(last_prices)
    portfolio_df['Current Value'] = portfolio_df['Current Price'] * portfolio_df['Quantity']
    total_portfolio_value = portfolio_df['Current Value'].sum()

    print(f"Total Portfolio Value: ${total_portfolio_value:,.2f}")
    print("\n--- Volatility Forecast (30-Day) ---")
    volatility_data = []
    for ticker in tickers:
        returns = stock_data['Close'][ticker].pct_change().dropna()
        forecasted_volatility = garch_volatility_forecast(returns)
        current_holding = portfolio_df[portfolio_df['Ticker'] == ticker]['Current Value'].iloc[0]
        volatility_data.append({
            'Ticker': ticker,
            'Current Holding': f"${current_holding:,.2f}",
            '30-Day Forecasted Volatility': f"{forecasted_volatility:.2f}%"
        })
    print(pd.DataFrame(volatility_data).to_string())

    print("\n--- Market Regime Analysis ---")
    market_index_data = get_stock_data('^GSPC', start_date, end_date)
    if not market_index_data.empty:
        regime_data = kmeans_regime_analysis(market_index_data)
        regime_returns = regime_data.groupby('Regime')['Close'].pct_change().dropna()
        avg_returns = regime_returns.groupby(regime_data['Regime']).mean()

        print("Average Daily Returns per Regime:")
        print(avg_returns.to_string())
    else:
        print("Warning: Could not download market index data for regime analysis.")

    print("\n--- News Analysis ---")
    for ticker in tickers:
        print(f'\nNews for {ticker}')
        try:
            sn = StockNews(ticker, save_news=False)
            df_news = sn.read_rss()
            for i in range(min(5, len(df_news))):
                print(f"  Title: {df_news['title'][i]}")
                print(f"  Published: {df_news['published'][i]}")
                print(f"  Summary: {df_news['summary'][i]}")
                print("---")
        except Exception as e:
            print(f"Could not fetch news for {ticker}: {e}")

if __name__ == "__main__":
    run_analysis()
