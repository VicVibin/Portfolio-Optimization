import numpy as np
from stochPathSim.heston_model import HestonModel
from stochPathSim.gbm import GeometricBrownianMotion
from stochPathSim.statistics_utils import calculate_statistics, display_statistics, plot_simulation


def option_pricer(mean, volatility, rf_rate, stock_price, strike_price, time, simulations, paths):
    disc = np.exp(-rf_rate * time)
    gbm = GeometricBrownianMotion(mu=mean, sigma=volatility, s0=stock_price, T=time, n=simulations, paths=paths)
    time, simulations = gbm.simulate()
    stats_df = calculate_statistics(simulations, stock_price)
    display_statistics(stats_df, time)
    plot_simulation(time, simulations, num_paths=200, xlabel="Time in years", ylabel="Stock Price")
    payoff = stats_df['Mean'][-1:].values
    option_price = max((payoff - strike_price) * disc,0)
    return option_price


mew = 0.2
sigma = 0.35
risk_free_rate = 0.04
stock = 300
strike = 301
time = 2
simulate = 1000
path = 2500

option_value = option_pricer(mew, sigma, risk_free_rate, stock, strike, time, simulate, path)
print(f"Call option Price: ${option_value}")