import yfinance as yf
import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from stochPathSim.heston_model import HestonModel
from stochPathSim.statistics_utils import calculate_statistics, display_statistics, plot_simulation
import statsmodels.api as sm
from fredapi import Fred

fred = Fred(api_key=str(input("Enter your api key: ")))
ten_year_treasury_note = fred.get_series_latest_release('GS10') / 100
risk_free_rate = ten_year_treasury_note.iloc[-1]


class OptionPricer(object):

    @staticmethod
    def JR_Method(initial_price, strike, time_to_maturity, sigma,  rate, time_step, option_type='P'):
        # precompute values
        dt = time_to_maturity / time_step
        nu = rate - 0.5 * sigma ** 2
        up = np.exp(nu * dt + sigma * np.sqrt(dt))
        down = 1 / up
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


class VolatilityModeling(object):

    @staticmethod
    # Download the historic volatility data to fit the model to
    def volatility(ticker, window, start, end):
        data = yf.download(ticker, start=start, end=end)['Close']
        initial_price = np.array(data)[-1]
        print(initial_price)
        log_returns = np.log(data / data.shift(1))
        volatility = log_returns.rolling(window=window).std() * np.sqrt(252)
        return log_returns[window:], volatility[window:], initial_price

    @staticmethod
    def mew_function(X, dt, kappa, theta):
        ekt = np.exp(-kappa * dt)
        return X * ekt + theta * (1 - ekt)

    @staticmethod
    def sigma_function(dt, kappa, sig):
        e2kt = np.exp(-2 * kappa * dt)
        return sig * np.sqrt((1 - e2kt) / (2 * kappa))

    def log_likelihood(self, theta_hat, X):
        kappa = theta_hat[0]
        theta = theta_hat[1]
        sig = theta_hat[2]
        X_dt = X[1:]
        X_t = X[:-1]
        dt = 1 / 252
        mew = self.mew_function(X_t, dt, kappa, theta)
        sigma = self.sigma_function(dt, kappa, sig)
        L_theta = np.sum(np.log(scipy.stats.norm.pdf(X_dt, loc=mew, scale=sigma)))
        return -L_theta

    @staticmethod
    def kappa_position(theta_hat):
        kappa = theta_hat[0]
        return kappa

    @staticmethod
    def theta_position(theta_hat):
        theta = theta_hat[1]
        return theta

    @staticmethod
    def sig_position(theta_hat):
        sig = theta_hat[2]
        return sig

    def optimize_params(self, ticker, window, start, end):
        log_returns, volatility, initial_price = self.volatility(ticker, window, start, end)
        vol = np.array(volatility)
        cons_set = [{'type': 'ineq', 'fun': self.kappa_position},
                    {'type': 'ineq', 'fun': self.sig_position}]
        theta_hat = [1, 1, 1]
        optimize = scipy.optimize.minimize(fun=self.log_likelihood,
                                           x0=theta_hat, args=vol, method='SLSQP', constraints=cons_set)
        K = optimize.x[0]
        theta = optimize.x[1]
        sig = optimize.x[2]

        return K, theta, sig, log_returns, vol, initial_price

    def Ornstein_Uhlenback_graph(self, K, theta, sig, vol, days):
        M = 1000
        Z = np.random.normal(size=M)
        drift = self.mew_function(vol[0], days, K, theta)
        diffusion = self.sigma_function(days, K, sig)
        X_t = drift + diffusion * Z
        plt.plot(X_t)
        plt.title('Ornstein-Uhlenbeck Process')
        plt.xlabel('Volatility')
        plt.show()

    def path_simulations(self, ticker, window, start, end, T, time_step, num_paths):
        K, theta, sig, log_returns, vol, initial_price = self.optimize_params(ticker, window, start, end)

        # Calculating rho, the correlation between the 2 brownian motions
        log_prices = log_returns.to_numpy()
        model = sm.OLS(log_prices, sm.add_constant(vol)).fit()

        # Initial Heston Model Parameters
        V0 = vol[-1]  # Initial Volatility
        rho = model.params[1]
        Heston_Simulation = HestonModel(V0, K, theta, sig, rho, initial_price, T, time_step, num_paths)
        stock_sims, volatility_sims = Heston_Simulation.simulate()
        stock_df = calculate_statistics(stock_sims, initial_price)
        time = [x for x in range(0, len(stock_df))]
        volatility_df = calculate_statistics(volatility_sims, V0)
        plot_simulation(time, stock_sims, num_paths=200, xlabel=f"Time step of {int(T * 365)} days",
                        ylabel="Stock Price")
        plot_simulation(time, volatility_sims, num_paths=200, xlabel=f"Time step of {int(T * 365)} days",
                        ylabel="Volatility")
        annualized_volatility = np.array(volatility_df["Mean"])[-1]
        return initial_price, annualized_volatility



ticker = "AAPL"
window = 100
start = "2023-01-01"
end = "2024-12-25"
T = 30/365
time_step = 5000
paths = 10000



initial_price, annualized_volatility = VolatilityModeling().path_simulations(ticker,window, start,
                                                                             end, T, time_step, paths)

call_price = OptionPricer().JR_Method(initial_price, 280, T, annualized_volatility,
                                      risk_free_rate, time_step, option_type="C")

print(f"${call_price:2f}")
