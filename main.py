from typing import Any
from sklearn.neural_network import MLPRegressor
import math
import scipy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from fredapi import Fred
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from stochPathSim.heston_model import HestonModel
import statsmodels.api as sm


# Linear Regression Model

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x).squeeze(1)


# StockDataset class
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Optimize_Heston:
    def __init__(self, dataframe, ticker):
        self.data = np.array(dataframe[ticker])
        self.returns = np.log(self.data[1:] / self.data[:-1])

        # Initialize parameters
        self.time = len(self.data)
        self.time_step = np.arange(self.time)
        self.S0 = self.data[0]
        self.dt = 1 / 252  # Daily data (252 trading days per year)

        # Initial guess based on historical data
        self.initial_mu = np.mean(self.returns) / self.dt + 0.5 * np.var(self.returns) / self.dt
        self.initial_sigma = np.std(self.returns) / np.sqrt(self.dt)
        self.last_price = self.data[-1]

    # Historical volatility of the data to fit the model to
    def volatility(self, window=10):
        stock_volatility = pd.Series(self.returns).rolling(window=window).std() * np.sqrt(252)
        return self.returns[window:], stock_volatility[window:], self.S0

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
        pdf = scipy.stats.norm.pdf(X_dt, loc=mew, scale=sigma)
        L_theta = np.sum(np.log(pdf))
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

    def optimize_params(self):
        log_returns, stock_volatility, initial_price = self.volatility()
        vol = np.array(stock_volatility)
        cons_set = [{'type': 'ineq', 'fun': self.kappa_position},
                    {'type': 'ineq', 'fun': self.sig_position}]
        theta_hat = [1, 1, 1]
        optimize = scipy.optimize.minimize(fun=self.log_likelihood,
                                           x0=theta_hat, args=vol, method='SLSQP', constraints=cons_set)
        K = optimize.x[0]
        theta = optimize.x[1]
        sig = optimize.x[2]

        return K, theta, sig, log_returns, vol, initial_price

    def future_returns(self, days_into_future):
        K, theta, sig, log_returns, vol, initial_price = self.optimize_params()

        # Calculating rho, the correlation between the 2 brownian motions
        model = sm.OLS(log_returns, sm.add_constant(vol)).fit()

        # Initial Heston Model Parameters
        V0 = vol[-1]  # Initial Volatility
        rho = model.params[1]
        Heston_Simulation = HestonModel(V0, K, theta, sig, rho, initial_price, 1, days_into_future, 10)
        stock_sims, volatility_sims = Heston_Simulation.simulate()
        stock_sims = stock_sims[0]
        returns: int | Any = (stock_sims[-1] / self.last_price) - 1
        return returns, stock_sims


class Optimize_GBM:
    def __init__(self, dataframe, ticker):
        # Download and process data
        self.data = np.array(dataframe[ticker])
        self.returns = np.log(self.data[1:] / self.data[:-1])

        # Initialize parameters
        self.time = len(self.data)
        self.time_step = np.arange(self.time)
        self.S0 = self.data[0]
        self.dt = 1 / 252  # Daily data (252 trading days per year)

        # Initial guess based on historical data
        self.initial_mu = np.mean(self.returns) / self.dt + 0.5 * np.var(self.returns) / self.dt
        self.initial_sigma = np.std(self.returns) / np.sqrt(self.dt)
        self.last_price = self.data[-1]

    def generate_gbm(self, S0, mu, sigma, time_steps):
        """
        Generate a GBM path with given parameters.
        """
        W = np.random.standard_normal(size=time_steps)
        t = np.arange(time_steps) * self.dt

        # More numerically stable implementation
        log_returns = (mu - 0.5 * sigma ** 2) * self.dt + sigma * np.sqrt(self.dt) * W
        log_prices = np.log(S0) + np.cumsum(log_returns)
        prices = np.exp(log_prices)
        prices = np.insert(prices, 0, S0)  # Add initial price

        return prices

    def objective(self, params):
        """
        Objective function to minimize (negative log-likelihood of the GBM model).
        """
        mu, sigma = params
        if sigma <= 0:
            return 1e10  # Penalty for invalid volatility

        # Calculate log-likelihood of observed returns
        log_likelihood = -np.sum(
            -0.5 * np.log(2 * np.pi) - np.log(sigma * np.sqrt(self.dt)) -
            (self.returns - (mu - 0.5 * sigma ** 2) * self.dt) ** 2 / (2 * sigma ** 2 * self.dt)
        )

        return log_likelihood

    def optimize_parameters(self):
        """
        Optimize GBM parameters using maximum likelihood estimation.
        """
        initial_guess = [self.initial_mu, self.initial_sigma]
        bounds = [(-1, 1), (0.0001, 1)]  # Reasonable bounds for μ and σ

        result = minimize(
            self.objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )

        return result.x

    def simulate(self):
        mu_opt, sig_opt = self.optimize_parameters()
        return mu_opt, sig_opt

    def simulate_and_plot(self, n_simulations=100):
        """
        Simulate GBM paths with optimized parameters and plot results.
        """
        mu_opt, sigma_opt = self.optimize_parameters()

        # Generate multiple paths
        plt.figure(figsize=(12, 8))
        plt.plot(self.time_step, self.data, 'b-', label='Historical Data', linewidth=2)

        for _ in range(n_simulations):
            simulated_prices = self.generate_gbm(self.S0, mu_opt, sigma_opt, self.time - 1)
            plt.plot(self.time_step, simulated_prices, 'r-', alpha=0.1)

        plt.plot([], [], 'r-', alpha=0.5, label='Simulated Paths')
        plt.xlabel('Trading Days')
        plt.ylabel('Stock Price')
        plt.title(f'GBM Simulation with μ={mu_opt:.4f}, σ={sigma_opt:.4f}')
        plt.legend()
        plt.grid(True)

        return mu_opt, sigma_opt

    def get_confidence_intervals(self, mu_opt, sigma_opt, confidence=0.95, n_simulations=10000):
        """
        Calculate confidence intervals for the price predictions.
        """
        simulations = np.zeros((n_simulations, self.time))

        for i in range(n_simulations):
            simulations[i] = self.generate_gbm(self.S0, mu_opt, sigma_opt, self.time - 1)

        lower_percentile = (1 - confidence) / 2
        upper_percentile = 1 - lower_percentile

        lower_bound = np.percentile(simulations, lower_percentile * 100, axis=0)
        upper_bound = np.percentile(simulations, upper_percentile * 100, axis=0)

        return lower_bound, upper_bound

    def future_returns(self, days_into_future=30):
        mu_opt, sig_opt = self.optimize_parameters()
        dataset = self.generate_gbm(self.last_price, mu_opt, sig_opt, days_into_future)
        returns = (dataset[-1] / self.last_price) - 1
        return returns, dataset


class StockOptim:
    def load_and_preprocess_data(self, n_tickers):
        fin_data = pd.read_csv('fin_data.csv', low_memory=False)  # reads CSV file
        unique_tickers = fin_data['symbol'].unique()  # defines unique tickers the ticker symbols
        viable_tickers = []  # creates an empty set of valid tickers to be later appended to
        # Relevant fields for determining the value of a stock
        required_fields = ['symbol', 'forwardPE', 'trailingPE', 'bookValue', 'freeCashflow', 'debtToEquity',
                           'returnOnAssets', 'returnOnEquity', 'beta', 'marketCap', 'dividendYield', 'priceToBook']

        # While loop to make sure my valid_tickers is greater than or equal to n tickers
        while len(viable_tickers) < n_tickers:
            random_tickers = np.random.choice(unique_tickers, size=50,
                                              replace=False)

            filtered_df = fin_data[fin_data['symbol'].isin(random_tickers)]
            # Handles edge cases if the required fields are not present in the CSV file for each ticker

            for ticker in random_tickers:
                # ticker data filters symbol and matches it to the ticker
                ticker_data = filtered_df[filtered_df['symbol'] == ticker]

                if not ticker_data[required_fields].isna().any().any():  # If an NA value isn't in the required fields
                    viable_tickers.append(ticker)  # Appends the ticker to the valid ticker list

            if len(viable_tickers) >= n_tickers:
                break  # Stops the while loop as soon as the viable_tickers is the size of the requested portfolio size

        viable_tickers = viable_tickers[:n_tickers]  # Viable tickers is the first n_tickers of the Viable_tickers list

        # Remove nan values
        viable_tickers = [x for x in viable_tickers if
                          not (isinstance(x, float) and math.isnan(x))]  # removes nan values

        adj_close_df = pd.DataFrame(yf.download(viable_tickers, start_date, end_date, threads= min(16, len(viable_tickers)))["Close"])
        adj_close_df = adj_close_df.dropna(axis=1, how="any")
        viable_tickers = adj_close_df.columns

        # Creates a data frame of all the stock symbols and the required fundamentals
        stock_fundamentals_df = fin_data[fin_data['symbol'].isin(viable_tickers)][required_fields]
        adj_close = adj_close_df.copy()

        #  Transposes the dataframe to make tickers in column 0 and the adjusted close prices in rows
        adj_close_df = pd.DataFrame(adj_close_df).T

        # Renames the column name symbol to ticker
        stock_fundamentals_df.rename(columns={'symbol': 'ticker'}, inplace=True)

        #  Sorts the data in alphabetical order
        adj_close_df = adj_close_df.sort_index()

        # Separates the ticker and date of column 0 so indexing and merging can take place
        adj_close_df = adj_close_df.reset_index()

        # Replaces all zeros with nan values in the dataset
        adj_close_df.replace(0, pd.NA, inplace=True)

        # renames Ticker as ticker for merging
        adj_close_df.rename(columns={'Ticker': 'ticker'}, inplace=True)

        # forward fills the 2nd row indexes to the last row to replace nan values
        cumprod_df = adj_close_df.iloc[:, 1:].ffill()

        # Calculates the percent change in the daily stock price
        returns_df = pd.DataFrame(cumprod_df.pct_change())

        # Calculates the cumulative product of the stock price to determine excess return
        cumulative_product = (1 + returns_df).cumprod()

        # Add the excess return as a new column to the DataFrame
        adj_close_df['excess_return'] = cumulative_product.iloc[:, -1]

        # sorts the stock fundamentals data by ticker in alphabetical order
        stock_fundamentals_df = stock_fundamentals_df.sort_values(by='ticker')
        # Merges the two datasets for machine learning and training
        merged_df = pd.merge(stock_fundamentals_df, adj_close_df, on='ticker', how='inner')

        return stock_fundamentals_df, merged_df, adj_close_df, adj_close

    def select_top_stocks(self, portfolio_size):
        while True:
            stock_fundamentals_df, merged_df, adj_close_df, adj_close = self.load_and_preprocess_data(
                max(portfolio_size * 2, 50))

            if len(merged_df['ticker'].unique()) < portfolio_size:
                print(f"Not enough stocks with complete data. Found {len(merged_df['ticker'].unique())}. Retrying.")
                continue

            # Feature engineering
            X = merged_df[['forwardPE', 'trailingPE', 'bookValue', 'freeCashflow', 'debtToEquity', 'returnOnAssets',
                           'returnOnEquity', 'beta', 'marketCap', 'dividendYield', 'priceToBook']]

            y = merged_df['excess_return']

            # Ensure X and y have the same number of samples
            X = X.loc[y.index]

            # Standardize features
            scaler = StandardScaler()
            X.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Drop rows with NaN values
            mask = ~X.isna().any(axis=1) & ~y.isna()
            X = X[mask]
            y = y[mask]

            X_scaled = scaler.fit_transform(X)

            # Define and train the model
            model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=0.0001, max_iter=50000)
            model.fit(X_scaled, y)

            # Predict future returns
            merged_df.loc[X.index, 'predicted_return'] = model.predict(X_scaled)

            # Select top stocks
            top_stocks = merged_df.groupby('ticker')['predicted_return'].mean().nlargest(portfolio_size).index.tolist()

            if len(top_stocks) == portfolio_size:
                return top_stocks, adj_close[top_stocks]
            else:
                print(
                    f"Found {len(top_stocks)} stocks. Retrying to match requested portfolio size of {portfolio_size}.")


class PortfolioOptim(object):

    @staticmethod
    def boy(tick, dataset, forward, model="GBM"):
        returns: list[int | Any] = []
        item_dataframe = pd.DataFrame()
        for t in tick:
            if model == "GBM":
                predictor = Optimize_GBM(dataset, t)
                item, item_GBM = predictor.future_returns(forward)
                returns.append(item)
                item_dataframe[t] = item_GBM
            else:
                predictor = Optimize_Heston(dataset, t)
                item, item_Heston = predictor.future_returns(forward)
                returns.append(item)
                item_dataframe[t] = item_Heston
        return returns, item_dataframe

    @staticmethod
    def expected_return(weights, computed_returns):
        portfolio_return = np.dot(computed_returns, weights)
        return portfolio_return

    #  Calculate standard deviation (volatility)
    @staticmethod
    def standard_deviation(weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)  # You don't need abs() for standard deviation, as variance is non-negative

    # Sharpe Ratio: (Expected Return - Risk-Free Rate) / Volatility
    def sharpe_ratio(self, weights, cov_matrix, fed_rate, precomputed_returns):
        port_return = self.expected_return(weights, precomputed_returns)
        port_volatility = self.standard_deviation(weights, cov_matrix)
        return (port_return - fed_rate) / port_volatility

    # Negative Sharpe Ratio: Minimizing negative Sharpe ratio is equivalent to maximizing Sharpe ratio

    def neg_sharpe_ratio(self, weights, cov_matrix, fed_rate, precomputed_returns):
        return -self.sharpe_ratio(weights, cov_matrix, fed_rate, precomputed_returns)

    def optimize_portfolio(self, tickers, returns_df, fed_rate):
        precomputed_returns, predicted_returns = self.boy(tickers, returns_df, V, Model)
        # Filter tickers based on the threshold of precomputed_returns
        threshold = 0.8  # Max precomputed return of a stock to ensure realism
        tickers, precomputed_returns = zip(*[(t, pr) for t, pr in zip(tickers, precomputed_returns) if pr <= threshold])

        # Filter the returns dataframe to keep only the selected tickers
        tickers, precomputed_returns = list(tickers), list(precomputed_returns)
        filtered_returns_df = returns_df[tickers]
        predicted_returns = predicted_returns[tickers]
        log_returns = np.log(filtered_returns_df / filtered_returns_df.shift(1))
        log_returns = log_returns.dropna()
        cov_matrix = log_returns.cov() * 252

        # Constraints: weights must sum to 1 (fully invested portfolio)
        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

        # Bounds for each weight: between 0 and 0.6 (for each ticker)
        bounds = [(0.0, 0.4) for _ in range(len(tickers))]

        # Initial guess for weights (equally distributed)
        initial_weights = np.array([1 / len(tickers)] * len(tickers))

        # Minimize the negative Sharpe ratio
        optimized_results = minimize(self.neg_sharpe_ratio, initial_weights,
                                     args=(cov_matrix, fed_rate, precomputed_returns),
                                     method='SLSQP', constraints=constraints, bounds=bounds)

        # Extract optimal weights
        optimal_weights = optimized_results.x

        # Calculate the optimal portfolio's return and volatility

        optimal_portfolio_return = self.expected_return(precomputed_returns, optimal_weights)
        optimal_portfolio_volatility = self.standard_deviation(optimal_weights, cov_matrix)
        optimal_sharpe_ratio = self.sharpe_ratio(optimal_weights, cov_matrix, fed_rate, precomputed_returns)
        return tickers, predicted_returns, optimal_weights, optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio


class IterativeOptim(object):
    def __init__(self):
        self.fred = fred
        self.ten_year_treasury_note = self.fred.get_series_latest_release('GS10') / 100
        self.risk_free_rate = self.ten_year_treasury_note.iloc[-1]
        self.rate = round(self.risk_free_rate * 100, 4)
        self.iteration_portfolio = 0
        self.best_sharpe_ratio = -np.inf
        self.best_portfolio = None

    def iteration(self):
        while self.iteration_portfolio < max_iterations_portfolio:
            self.iteration_portfolio += 1
            print(f"\nIteration {self.iteration_portfolio} of {max_iterations_portfolio}")
            optimized_stock = set()
            returns_dataset = pd.DataFrame()
            while len(optimized_stock) < portfolio_size:
                new_optimized_stock, returns = StockOptim().select_top_stocks(portfolio_size)
                optimized_stock.update(new_optimized_stock)
                returns_dataset = pd.concat([returns_dataset, returns], axis=1)
            sorted_optimized_stock = sorted(optimized_stock)
            valid_tickers = sorted_optimized_stock
            valid_tickers, pr_r, pw, pr, pv, sr = PortfolioOptim().optimize_portfolio(valid_tickers, returns_dataset,
                                                                                      self.risk_free_rate)
            print(f"Expected Return for {len(returns_dataset)} holding days: {pr * 100:.4f}%")
            print(f"Expected Annualized holding Return: {(((1 + pr) ** (252 / V)) - 1) * 100:.2f}%")
            print(f"Expected Volatility: {pv * 100:.4f}%")
            print(f"Sharpe Ratio: {sr:.4f}")
            if self.best_sharpe_ratio < sr < 4:
                self.best_sharpe_ratio = sr
                self.best_portfolio = (valid_tickers, pr_r, pw, pr, pv, sr)
            if target_sharpe_ratio <= sr < 4 and 1000 >= pr >= target_ear and pv <= target_volatility:
                print("Target Sharpe Ratio and Expected Annual Return met!")
                break
            else:
                print("Target not met. Retrying with a new set of stocks.")
        if self.iteration_portfolio == max_iterations_portfolio:
            print("\nMaximum iterations reached. Using the best portfolio found")
            valid_tickers, pr_r, pw, pr, pv, sr = self.best_portfolio
            print(f"Best Sharpe Ratio: {sr:.4f}")
            print(f"Best Expected Annual Return: {pr * 100:.4f}%")
            print(f"Best Expected Volatility: {pv * 100:.4f}%")
        else:
            print("\nOptimal portfolio found:")

        new_valid_tickers = []
        new_optimal_weights0 = []
        new_pr_r = []

        for i in range(len(pw)):
            if abs(pw[i]) >= 0.0001:
                new_valid_tickers.append(valid_tickers[i])
                new_optimal_weights0.append(pw[i])
        for tick in new_valid_tickers:
            new_pr_r.append(pr_r[tick])

        new_pr_r = pd.DataFrame(new_pr_r).T
        print(f"Optimal Portfolio size: {len(new_valid_tickers)}")

        print("\nFinal Optimal Weights:")
        for ticker, weight in zip(new_valid_tickers, new_optimal_weights0):
            print(f"{ticker}: {weight:.4f}")
        new_optimal_weights = np.array(new_optimal_weights0)
        plt.figure(figsize=(50, 50))
        plt.bar(new_valid_tickers, new_optimal_weights)
        plt.xlabel('Assets')
        plt.ylabel('Optimal Weights')
        plt.title('Optimal Portfolio Weights')
        plt.show()
        return new_valid_tickers, new_pr_r, new_optimal_weights0, pr, pv, sr


class CapAllocation():
    def __init__(self):
        self.fred = fred
        self.ten_year_treasury_note = self.fred.get_series_latest_release('GS10') / 100
        self.risk_free_rate = self.ten_year_treasury_note.iloc[-1]
        self.rate = round(self.risk_free_rate * 100, 4)

    def capital_allocation(self, tickers, weights, t):
        weights = np.array(weights)
        print("Capital Allocation for Initial Backtest Investment Amount")
        for ticker, weight in zip(tickers, weights):
            print(
                f"{ticker}: {round(t * weight, 2)} : Company Name: {yf.Ticker(ticker).info['longName']}, # of Shares :"
                f"{round(float(round(t * weight, 2) / float(yf.Ticker(ticker).info['currentPrice'])), 2)}")
        return tickers, weights

    def Backtest(self, tickers, pr_r, weights, pr, pv, t):
        data = yf.download(tickers, start=start_date2, end=end_date2)['Close']
        returns = data.pct_change()
        predicted_returns = pr_r.pct_change()
        p_portfolio_returns = predicted_returns.dot(weights)
        portfolio_returns = returns.dot(weights)
        p_cum_returns = t * (1 + p_portfolio_returns).cumprod()
        cumulative_returns = t * (1 + portfolio_returns).cumprod()
        p_cum_returns = p_cum_returns[:len(cumulative_returns)]
        total_return = cumulative_returns.iloc[-1]
        annualized_return = (total_return / t) ** (252 / len(data)) - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)  # Assuming 252 trading days in a year
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        equity_curve = pd.Series(cumulative_returns)

        # Calculate rolling maximum (peak)
        rolling_max = equity_curve.cummax()

        # Drawdown calculation
        drawdown = (equity_curve - rolling_max) / rolling_max

        # Maximum Drawdown
        max_drawdown = drawdown.min()  # Since drawdowns are negative, min() gives the worst drop
        calmar_ratio = annualized_return / -max_drawdown

        # Display results

        print(f"BACKTEST RESULTS")
        print(f"Trading Days: {len(data)}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Volatility: {annualized_volatility:.2%}")
        print(f"Predicted Total Returns: ${round(p_cum_returns.iloc[-1], 2)}")
        print(f"Total Returns: ${total_return:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Calmar Ratio: {calmar_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown * -100:.2f}%")

        # Plot equity and drawdown
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(equity_curve, label='Equity Curve')
        plt.plot(rolling_max, linestyle='--', label='Rolling Max')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.fill_between(drawdown.index, drawdown, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.tight_layout()
        plt.show()

        if stop_loss_strategy == "Y":
            predicted_value = t * (1 + pr)
            p_returns = []
            # Check for the condition in daily returns
            for i in range(1, len(cumulative_returns)):
                portfolio_value = cumulative_returns.iloc[i]
                p_returns.append(portfolio_value)

                if portfolio_value >= predicted_value:
                    cash_returned = pr
                    deviation = np.array(p_returns).std() * np.sqrt(len(p_returns))
                    print(f"Actual amount returned after sell trigger: {cash_returned * 100:.2f}")
                    print(f"Sell triggered on day {len(p_returns)} with portfolio value: ${portfolio_value:.2f}")
                    break
        else:

            # Download data for the train data
            data_train = yf.download(tickers, start=datetime.today() - timedelta(days=3 * 365),
                                     end=datetime.today() - timedelta(days=365))['Close']
            returns_train = data_train.pct_change(fill_method=None).dropna()
            portfolio_returns_train = returns_train.dot(weights)
            cumulative_returns_train = (1 + portfolio_returns_train).cumprod()

            data_test = yf.download(tickers, start=datetime.today() - timedelta(days=365),
                                    end=end_date2)['Close']
            returns_test = data_test.pct_change(fill_method=None).dropna()
            portfolio_returns_test = returns_test.dot(weights)
            cumulative_returns_test = (cumulative_returns_train).iloc[-1] * ((1 + portfolio_returns_test).cumprod())

            # Split data into train and test
            train_data = cumulative_returns_train.index
            test_data = cumulative_returns_test.index

            # Prepare X values for training and testing
            X_train = np.arange(len(train_data)).reshape(-1, 1)
            X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)

            # Normalize data
            x_mean, x_std = X_train.mean(), X_train.std()
            X_train_normalized = (X_train - x_mean) / x_std
            X_test_normalized = (X_test - x_mean) / x_std
            y_mean, y_std = cumulative_returns_train.mean(), cumulative_returns_train.std()
            y_train_normalized = (cumulative_returns_train.values - y_mean) / y_std

            # Convert to tensors
            X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_normalized, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)

            # Initialize and train the model
            model = LinearRegressionModel(1, 1)
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001)
            num_loops = 3000

            for loop in range(num_loops):
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Make predictions
            model.eval()
            with torch.no_grad():
                train_prediction_normalized = model(X_train_tensor)
                test_prediction_normalized = model(X_test_tensor)

            # Denormalize predictions
            train_prediction = train_prediction_normalized.numpy() * y_std + y_mean
            test_prediction = test_prediction_normalized.numpy() * y_std + y_mean

            confidence = 3

            # Get the slope and intercept from the trained model
            weight = model.linear.weight.item()  # Slope (weight)

            # Adjusted parameters
            adjusted_slope = weight * (y_std / x_std)

            adjusted_intercept = cumulative_returns_train.iloc[-1]
            # Create X based on the number of test predictions
            elem = np.arange(len(test_prediction))  # Create an array from 0 to len(test_prediction)
            # Compute bounds based on the modified X
            bounds = confidence * annualized_volatility / 100 * np.sqrt(
                elem + (1 / (elem + 1)) + (confidence / (annualized_volatility / 100)))
            y_upper = adjusted_slope * elem + adjusted_intercept + bounds
            y_lower = adjusted_slope * elem + adjusted_intercept - bounds

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(train_data, cumulative_returns_train.values, label='Training Data', color='blue')
            plt.plot(test_data, cumulative_returns_test.values, label='Test Data', color='orange')
            plt.plot(test_data, test_prediction, label='Model Prediction (Test)', color='green')
            plt.plot(test_data, y_upper, label='Upper Bound', color='green', linestyle='--')
            plt.plot(test_data, y_lower, label='Lower Bound', color='red', linestyle='--')
            plt.title('Portfolio Cumulative Returns and Linear Regression Prediction')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid(True)
            plt.show()

        if annualized_volatility - pv > 0:
            print(f"Volatility underestimated by {abs(annualized_volatility - pv):.2%}")
        else:
            print(f"Volatility overstated by {abs(annualized_volatility - pv):.2%}")

        if annualized_volatility < pv:
            print(f" Win")
        else:
            print(f"Loss")

        if annualized_return - pr > 0:
            print(f" Returns underestimated by {abs(annualized_return - pr):.2%}")
        else:
            print(f" Returns overstated by {abs(annualized_return - pr):.2%}")

        if annualized_return > pr:
            print(f" Win")
        else:
            print(f"Loss")

        if total_return - t >= 0:
            print(f"Gain of ${round(total_return - t, 2)} on ${t} investment")
        else:
            print(f"Loss of ${round(t - total_return, 2)} on ${t} investment")

        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns.values, color="g", label="Actual")
        plt.plot(cumulative_returns.index, p_cum_returns.values, color="r", label="Predicted")
        plt.title('Portfolio Cumulative Returns (Predicted VS Actual)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.show()


fred = fred = Fred(api_key=str(input("Enter your api key: ")))
ten_year_treasury_note = fred.get_series_latest_release('GS10') / 100
risk_free_rate = ten_year_treasury_note.iloc[-1]

# Input parameters
t = int(input("Enter your initial portfolio optimizer end year:"))
t_1 = int(input("Enter the length of time data for optimization:"))
V = int(input("Enter the length of portfolio hold period in days: "))
portfolio_size = int(input("Enter your initial requested portfolio size:"))  # 50 = optimal portfolio size
target_sharpe_ratio = float(input("Enter your target Sharpe ratio:"))
target_volatility = float(input("Enter your target volatility (%):")) / 100
target_ear = float(input("Enter your target Expected Annual Return (%):")) / 100  # Convert to decimal
max_iterations_portfolio =  int(input("Enter your max number of iterations:"))
stop_loss_strategy = str(input("Maxed_Expectation:"))
end_date = datetime.today() - timedelta(days=int(t * 365))
start_date = end_date - timedelta(days=int(t_1 * 365))
Model = str(input("Pick Underlying Stock Model:"))
S = int(365 * V / 252)
# Backtest dates
start_date2 = end_date.strftime('%Y-%m-%d')
end_date2 = (end_date + timedelta(days=S)).strftime('%Y-%m-%d')

stocks, predicted_GBM, ratios, returns, volatility, sharpe = IterativeOptim().iteration()
t = float(input("Enter your initial backtest investment amount:"))

try:
    _, _ = CapAllocation().capital_allocation(stocks, ratios, t)
except:
    C = 0
CapAllocation().Backtest(stocks, predicted_GBM, ratios, returns, volatility, t)
