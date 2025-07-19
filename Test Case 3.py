import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.stattools import coint
from datetime import datetime, timedelta
import concurrent.futures


## Attempt Threadpooling

class PairsTrading(object):
    def pairsTrading(self, dollars, leverage, historic_window, ticker1, ticker2, start, start_date, end_date):
        h1, h2, coint_data1, coint_data2, boolean, tickers = self.download_data(historic_window, leverage, ticker1,
                                                                                ticker2, start, start_date, end_date)
        hedge_ratio, spread, boolean = self.cointegration(h1, h2, coint_data1, coint_data2, boolean)
        spread, long_signal, short_signal, exit_signal, boolean, multiplier = self.stationary_time_series(dollars, spread, boolean)
        indicator, boolean = self.indicators(spread, long_signal, short_signal, exit_signal, boolean)
        pnl, number_trades = self.PNL(dollars, tickers, hedge_ratio, indicator, coint_data1, coint_data2)
        return pnl, number_trades

    @staticmethod
    def PNL(dollars, tickers, hedge_ratio, indicator: list, data1, data2):
        indicator[-1] = "Exit"
        pnl = 0
        funding_cost = 0
        historic_position = 0
        position_value_array = []
        num_of_trades = 0
        for i in range(len(indicator)):
            if indicator[i] == "Long":
                position_value_array.append([dollars * data1[i], -dollars * hedge_ratio * data2[i]])
                num_of_trades += 1
                historic_position = "Long"
            elif indicator[i] == "Short":
                position_value_array.append([-dollars * data1[i], dollars * hedge_ratio * data2[i]])
                historic_position = "Short"
                num_of_trades += 1
            if indicator[i] == "Exit" and historic_position == "Long":
                pnl += ((dollars * (data1[i] - position_value_array[0][0])) -
                        (dollars * hedge_ratio * (data2[i] - position_value_array[0][1])))
                position_value_array = []
                historic_position = "Exit"
                num_of_trades += 1
            elif indicator[i] == "Exit" and historic_position == "Short":
                pnl += ((dollars * (data1[i] - position_value_array[0][0])) -
                        (dollars * hedge_ratio * (data2[i] - position_value_array[0][1])))
                position_value_array = []
                historic_position = "Exit"
                num_of_trades += 1

        long_normalizer = (data1[-1] + data2[-1] - data1[0] - data2[0]) * dollars
        print(f"Cointegrated tickers: {tickers}")
        print(f"Long holdings return: {long_normalizer}")
        print(f"PNL: ${pnl:.2f}")
        print(f"PTL: {round(pnl / long_normalizer,2)}x")
        print(f"Number of trades: {num_of_trades}")
        return pnl, num_of_trades

    @staticmethod
    def download_data(historic_window, leverage, ticker1, ticker2, start, start_date, end_date):
        boolean = True
        tickers = [ticker1, ticker2]
        historic_data1 = np.array(yf.download(ticker1, start=start, end=start_date)["Adj Close"]) * leverage
        historic_data2 = np.array(yf.download(ticker2, start=start, end=start_date)["Adj Close"]) * leverage
        historic_data1 = historic_data1[-historic_window:]
        historic_data2 = historic_data2[:len(historic_data1)]
        if len(historic_data1) == 0 or len(historic_data2) == 0:
            boolean = False
        data1 = np.array(yf.download(ticker1, start=start_date, end=end_date)["Adj Close"]) * leverage
        data2 = np.array(yf.download(ticker2, start=start_date, end=end_date)["Adj Close"]) * leverage
        if np.nan in data1 or np.nan in data2:
            return historic_data1, historic_data2, data1, data2, boolean
        else:
            coint_data1 = data1
            coint_data2 = data2[:len(coint_data1)]
            coint_data1 = coint_data1[:len(coint_data2)]
            print(len(coint_data1), len(coint_data2))
        return historic_data1, historic_data2, coint_data1, coint_data2, boolean, tickers

    @staticmethod
    def cointegration(h1, h2, coint_data1, coint_data2, boolean):
        if boolean:
            model = sm.OLS(h1, sm.add_constant(h2)).fit()
            if len(model.params) > 1:
                hedge_ratio = model.params[1]
                print(f" Hedge Ratio: {hedge_ratio}")
                spread = coint_data1 - hedge_ratio * coint_data2
                return hedge_ratio, spread, boolean
            else:
                return False, [], False
        else:
            return False, [], False

    @staticmethod
    def stationary_time_series(dollars, historic_spread, boolean):
        if len(historic_spread) == 0:
            return False, False, False, False, False
        mean = np.mean(historic_spread)
        std = np.std(historic_spread)
        multiplier = dollars / (abs(mean))

        # Define thresholds
        upper_threshold = mean + std
        lower_threshold = mean - std

        # Generate signals
        long_signal = historic_spread < lower_threshold  # Go long the spread
        short_signal = historic_spread > upper_threshold  # Go short the spread
        exit_signal = (historic_spread >= mean) | (historic_spread <= mean)  # Exit when spread reverts

        # Initialize variables
        # Plot the spread
        plt.plot(historic_spread)
        plt.axhline(np.mean(historic_spread), color='red', linestyle='--', label='Mean')
        plt.axhline(np.mean(historic_spread) + 1 * np.std(historic_spread), color='green', linestyle='--',
                    label='Mean + 1 Std')
        plt.axhline(np.mean(historic_spread) - 1 * np.std(historic_spread), color='green', linestyle='--',
                    label='Mean - 1 Std')
        plt.legend()
        plt.show()
        return historic_spread, long_signal, short_signal, exit_signal, boolean, multiplier

    @staticmethod
    def indicators(spread, long_signal, short_signal, exit_signal, boolean):
        positions = np.zeros(len(spread))  # 1 for long, -1 for short, 0 for no position
        # Generate positions
        for i in range(1, len(spread)):
            if long_signal[i]:
                positions[i] = 1  # Long the spread
            elif short_signal[i]:
                positions[i] = -1  # Short the spread
            elif exit_signal[i]:
                positions[i] = 0  # Exit position
        data = positions
        indicator = []
        historic_position = 0
        for values in range(len(data)):
            if data[values] == 1 and historic_position != data[values]:
                indicator.append("Long")
                historic_position = 1
            elif data[values] == -1 and historic_position != data[values]:
                indicator.append("Short")
                historic_position = -1
            elif data[values] == 0 and historic_position != data[values]:
                indicator.append("Exit")
                historic_position = 0
            else:
                indicator.append(0)
        return indicator, boolean


class PairsFinder(object):

    @staticmethod
    def load_and_preprocess_data(trading_window, pairs: int, start_date, end_date):
        df = pd.read_csv('fin_data.csv', low_memory=False)
        viable_pairs = []
        p_values = []
        while len(viable_pairs) <= pairs:
            for sector, group in df.groupby('industry'):
                sector_length = len(df[df['sector'] == sector])
                tickers = group['symbol'].head(max(sector_length, 50)).tolist()
                adj_close = pd.DataFrame(yf.download(tickers, start_date, end_date)['Adj Close'][-trading_window:])
                adj_close = adj_close.dropna(axis=1)
                columns = adj_close.columns
                for i in range(len(columns) - 1):
                    coint_data1 = adj_close[columns[i]]
                    if len(viable_pairs) == pairs:
                        return viable_pairs, p_values
                    for j in range(i + 1, len(columns)):
                        if len(viable_pairs) == pairs:
                            return viable_pairs, p_values
                        coint_data2 = adj_close[columns[j]][:len(coint_data1)]
                        if np.var(coint_data1) == 0 or np.var(coint_data2) == 0:
                            continue
                        score, p_value, lag = coint(coint_data1, coint_data2)
                        if p_value < 0.001:
                            p_values.append(p_value)
                            viable_pairs.append([columns[i], columns[j]])
        return viable_pairs, p_values


    @staticmethod
    def process_sector(sector_data, trading_window, start_date, end_date, pairs):
        """
        Processes a single sector: downloads data and identifies viable pairs.
        """
        viable_pairs = []
        p_values = []

        # Get tickers and download Adjusted Close data
        tickers = sector_data['symbol'].tolist()
        adj_close = pd.DataFrame(yf.download(tickers, start=start_date, end=end_date)['Adj Close'][-trading_window:])
        adj_close = adj_close.dropna(axis=1)  # Drop columns with NaNs
        columns = adj_close.columns

        # Parallelize cointegration testing
        def test_pair(i, j):
            coint_data1 = adj_close[columns[i]]
            coint_data2 = adj_close[columns[j]][:len(coint_data1)]
            if np.var(coint_data1) == 0 or np.var(coint_data2) == 0:
                return None, None
            score, p_value, lag = coint(coint_data1, coint_data2)
            if p_value < 0.001:
                return (columns[i], columns[j]), p_value
            return None, None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(test_pair, i, j)
                for i in range(len(columns) - 1)
                for j in range(i + 1, len(columns))
            ]
            for future in futures:
                pair, p_value = future.result()
                if pair and len(viable_pairs) < pairs:
                    viable_pairs.append(pair)
                    p_values.append(p_value)
                if len(viable_pairs) >= pairs:
                    break

        return viable_pairs, p_values

    def load_and_preprocess(self, trading_window, pairs, start_date, end_date):
        df = pd.read_csv('fin_data.csv', low_memory=False)
        results = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            sector_futures = {
                executor.submit(self.process_sector, group, trading_window, start_date, end_date, pairs): sector
                for sector, group in df.groupby('sector')
            }
            for future in sector_futures:
                viable_pairs, p_values = future.result()
                results.extend(zip(viable_pairs, p_values))
                if len(results) >= pairs:
                    break

        # Combine results and limit to the requested number of pairs
        viable_pairs, p_values = zip(*results[:pairs])
        return list(viable_pairs), list(p_values)


def tradingAlgo(dollars, leverage, pairs_number, historic_window, cointegration_window, end_date):
    end = end_date - timedelta(days=cointegration_window)
    start = end - timedelta(days=5 * 365)
    viable_pairs, p_values = PairsFinder().load_and_preprocess_data(historic_window, pairs_number, start, end)
    equity_list = []
    count = 0
    for p_val in p_values:
        count += 1 / p_val
    for p_val in p_values:
        equity_list.append((1 / p_val) / count)
    equity_list = dollars * np.array(equity_list)
    accumulated_pnl = 0
    total_trades = 0
    for i in range(len(viable_pairs)):
        pnl, trades = PairsTrading().pairsTrading(dollars, leverage, historic_window, viable_pairs[i][0], viable_pairs[i][1],
                                                  start, end, end_date)
        accumulated_pnl += pnl
        total_trades += trades
    print(f" Total accumulated PNL from trading {pairs_number} pairs of stocks: ${accumulated_pnl:.2f}")
    print(f" Total trades: {total_trades}")
    return True



def currentCointegratedStocks(start, pairs):
    end = datetime.today()
    viable_pairs, p_values = PairsFinder().load_and_preprocess_data(-0, pairs, start, end)
    return viable_pairs, p_values


contracts = 1
total_traded_equity = 1000
pairs = 40
historic_window = 300
trading_window = 365
end_2 = datetime.today()-timedelta(days=0* 365)

tradingAlgo(total_traded_equity, contracts, pairs, historic_window, trading_window, end_2)

