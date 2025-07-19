import yfinance as yf
import numpy as np
import scipy
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize


class OU_optimization(object):
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

    def OU_optimize(self, series):
        series = np.array(series)
        cons_set = [{'type': 'ineq', 'fun': self.kappa_position},
                    {'type': 'ineq', 'fun': self.sig_position}]
        theta_hat = [1, 1, 1]
        optimize = scipy.optimize.minimize(fun=self.log_likelihood,
                                           x0=theta_hat, args=series, method='SLSQP', constraints=cons_set)
        K = optimize.x[0]
        theta = optimize.x[1]
        sig = optimize.x[2]

        return K, theta, sig, series[-1]


class StockModeling(object):
    @staticmethod
    # Download the historic volatility data to fit the model to
    def beta(ticker, window, start, end):
        market_data = np.array(yf.download("^GSPC", start=start, end=end)["Close"])
        stock_data = np.array(yf.download(ticker, start=start, end=end)["Close"])
        scaled_market_data = StandardScaler().fit_transform(market_data.reshape(-1, 1))
        scaled_stock_data = StandardScaler().fit_transform(stock_data.reshape(-1, 1))
        rolling_beta = []
        for i in range(window, len(scaled_market_data)):
            X = scaled_market_data[i - window:i]
            y = scaled_stock_data[i - window:i]
            X = sm.add_constant(X)
            model = OLS(y, X).fit()
            beta = model.params[1]
            rolling_beta.append(beta)
        return rolling_beta

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
        rolling_beta = self.beta(ticker, window, start, end)
        betas = np.array(rolling_beta)
        cons_set = [{'type': 'ineq', 'fun': self.kappa_position},
                    {'type': 'ineq', 'fun': self.sig_position}]
        theta_hat = [1, 1, 1]
        optimize = scipy.optimize.minimize(fun=self.log_likelihood,
                                           x0=theta_hat, args=betas, method='SLSQP', constraints=cons_set)
        K = optimize.x[0]
        theta = optimize.x[1]
        sig = optimize.x[2]

        return K, theta, sig, betas[-1]

    def OU_graph(self, K, theta, sig, beta, days, samples):
        days = days / 365
        M = samples
        Z = np.random.normal(size=M)
        drift = self.mew_function(beta, days, K, theta)
        diffusion = self.sigma_function(days, K, sig)
        X_t = drift + diffusion * Z
        plt.plot(X_t)
        plt.title('Ornstein-Uhlenbeck Process')
        plt.xlabel('Beta')
        plt.show()

    def beta_modeling(self, K, theta, sig, beta, days, samples):
        days = days / 252
        M = samples
        Z = np.random.normal(size=M)
        drift = self.mew_function(beta, days, K, theta)
        diffusion = self.sigma_function(days, K, sig)
        X_t = drift + diffusion * Z
        return X_t


    def returns(self, ticker, start, end, window, samples, rf_rate, market_rate):
        rf_rate, market_rate = rf_rate/252, market_rate/252
        dt = 252 / samples
        K, theta, sigma, initial_value = StockModeling().optimize_params(ticker, window, start, end)
        betas = self.beta_modeling(K, theta, sigma, initial_value, 252, samples)
        value = 0
        premium = market_rate - rf_rate
        for beta in betas:
            value += rf_rate * dt + beta * premium * dt
        return value

returns = StockModeling().returns("META", "2018-01-01", "2024-01-01", 30, 5000,
                                  0.04, 0.25)
print(returns)