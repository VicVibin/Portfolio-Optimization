import yfinance as yf
import pandas as pd
from datetime import timedelta, datetime
import numpy as np
from fredapi import Fred

fred = Fred(api_key=str(input("Enter your api key: ")))
ten_year_treasury_note = fred.get_series_latest_release('GS10') / 100
risk_free_rate = ten_year_treasury_note.iloc[-1]


def jr_method(strike, time_to_maturity, initial_price, rate, time_step, sigma, option_type='P'):
    # precompute values
    dt = time_to_maturity / time_step
    nu = rate - 0.5 * sigma ** 2
    up = np.exp(nu * dt + sigma * np.sqrt(dt))
    down = np.exp(nu * dt - sigma * np.sqrt(dt))
    q = (np.exp(rate * dt) - down) / (up - down)
    disc = np.exp(-rate * dt)

    # initialise stock prices at maturity
    S = initial_price * down ** (np.arange(time_step, -1, -1)) * up ** (np.arange(0, time_step + 1, 1))

    # option payoff
    if option_type == 'P':
        C = np.maximum(0, strike - S)
    else:
        C = np.maximum(0, S - strike)

    # backward recursion through the tree
    for i in np.arange(time_step - 1, -1, -1):
        S = initial_price * down ** (np.arange(i, -1, -1)) * up ** (np.arange(0, i + 1, 1))
        C[:i + 1] = disc * (q * C[1:i + 2] + (1 - q) * C[0:i + 1])
        C = C[:-1]
        if option_type == 'P':
            C = np.maximum(C, strike - S)
        else:
            C = np.maximum(C, S - strike)

    return C[0]


# Define the ticker symbol
# Fetch data for the ticker symbol

ticker_symbol = 'NVDA'
volatility = yf.download(ticker_symbol,
                         start=datetime.today() - timedelta(days=1 * 365),
                         end=datetime.today())['Adj Close'].pct_change().std() * np.sqrt(252)

stock = yf.Ticker(ticker_symbol)
stock_price = yf.download(ticker_symbol, start=datetime.today() - timedelta(days=1), end=datetime.today())['Adj Close']
# Get the expiration dates for the options available
expiration_dates = stock.options
print(f"Expiration dates: {expiration_dates}")

# Fetch the option chain for a specific expiration date
expiration_date = expiration_dates[0]  # Pick the first expiration date
print(f"time_to_maturity: {(datetime.strptime(expiration_date, "%Y-%m-%d") - datetime.today()).days / 365.25}")
option_chain = stock.option_chain(expiration_date)
option_chain_calls = pd.DataFrame(option_chain.calls)
option_chain_puts = pd.DataFrame(option_chain.puts)

for i in range(len(option_chain_calls)):
    strike = option_chain_calls['strike'][i]
    sigma_HV = volatility
    time_step = 1000
    initial_price = stock_price.iloc[0]
    rate = risk_free_rate
    time_to_maturity = (datetime.strptime(expiration_date, "%Y-%m-%d") - datetime.today()).days / 365.25
    binomial_tree_price_historic = round(jr_method(strike, time_to_maturity,
                                                   initial_price, rate, time_step, sigma_HV, option_type='C'), 2)

    print(f"HV: {sigma_HV:.2f}, "
          f"Bid Price:{option_chain_calls['bid'][i]}, Ask Price:{option_chain_calls['ask'][i]} "
          f"HV price: {binomial_tree_price_historic:.2f} "
          f"Strike Price: {option_chain_calls['strike'][i]}, Stock Price: {stock_price.iloc[0]}")
    if binomial_tree_price_historic < 0.01:
        binomial_tree_price_historic = 0.01
    option_chain_calls.loc[i, "Binomial Tree Price_HV"] = binomial_tree_price_historic

# Option chain data contains two parts: calls and puts
print(f"Call options data for expiration date:  {expiration_date}:")
print(f"Stock Price as of today:{stock_price.iloc[0]}")
# Display call options
print(option_chain_calls[['impliedVolatility', 'strike', 'bid', 'ask',
                          'lastPrice', 'Binomial Tree Price_HV', "Risk Insurance"]])
