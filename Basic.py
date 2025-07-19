# Stock Model #10
# Implemented EX return lstm model
# Implemented a basic sell strategy
from sklearn.neural_network import MLPRegressor
import math
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
from torch.utils.data import Dataset, DataLoader

fred = Fred(api_key='c496761920d5eedc780bf493ac3ae0e9')
ten_year_treasury_note = fred.get_series_latest_release('GS10') / 100
risk_free_rate = ten_year_treasury_note.iloc[-1]

# Input parameters
t = int(input("Enter your initial portfolio optimizer end year:"))
t_1 = int(input("Enter the length of time data for optimization:"))
portfolio_size = int(input("Enter your initial requested portfolio size:"))  # 50 = optimal portfolio size
target_sharpe_ratio = float(input("Enter your target Sharpe ratio:"))
target_volatility = float(input("Enter your target volatility (%):")) / 100
target_ear = float(input("Enter your target Expected Annual Return (%):")) / 100  # Convert to decimal
max_iterations_portfolio = int(input("Enter your max number of iterations:"))
end_date = datetime.today() - timedelta(days=t * 365)
start_date = end_date - timedelta(days=t_1 * 365)

# Backtest dates
start_date2 = end_date.strftime('%Y-%m-%d')
end_date2 = (end_date + timedelta(days=365)).strftime('%Y-%m-%d')


# StockDataset class
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# PyTorch LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1=100, hidden_size2=50, dense_size=25):
        super(StockLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(hidden_size2, dense_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(dense_size, 1)

    def forward(self, x):
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)

        # We only want the last output for the dense layers
        dense_input = lstm2_out[:, -1, :]

        # Dense layers
        dense1_out = self.relu(self.dense1(dense_input))
        output = self.dense2(dense1_out)

        return output


# Create sequences
def create_sequences(X, y, look):
    Xs, ys = [], []
    for i in range(len(X) - look):
        Xs.append(X[i:(i + look)])
        ys.append(y[i + look - 1])
    return np.array(Xs), np.array(ys)


# Expected Return function
def expectations(t, start, end, look=60):
    # Download stock data
    print(f" Ticker being analyzed {t}")
    data = yf.download(t, start=start, end=end)
    data['Returns'] = data['Adj Close'].pct_change()
    data['MA50'] = data['Adj Close'].rolling(window=50).mean()
    data['MA200'] = data['Adj Close'].rolling(window=200).mean()
    data['Volatility'] = data['Returns'].rolling(window=50).std()
    data.dropna(inplace=True)
    data['Target'] = data['Adj Close'].shift(-252) / data['Adj Close'] - 1
    data = data[:-252]  # Remove last 252 rows for prediction

    features = ['Adj Close', 'Returns', 'MA50', 'MA200', 'Volatility']
    X = data[features].values
    y = data['Target'].values

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y, look)

    # Split data into training and testing
    train_size = int(len(X_seq) * 0.8)
    X_train = X_seq[:train_size]
    X_test = X_seq[train_size:]
    y_train = y_seq[:train_size]
    y_test = y_seq[train_size:]

    # Create datasets and dataloaders
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    if torch.cuda.is_available():
        print("Running on GPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockLSTM(input_size=X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 500
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs.squeeze(), batch_y).item()

        train_loss /= len(train_loader)
        val_loss /= len(test_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    # Load best model
    model.load_state_dict(best_model_state)

    # Save the model
    torch.save(model.state_dict(), 'stock_lstm.pth')

    # Latest prediction
    latest_data = torch.FloatTensor(X_scaled[-look:]).unsqueeze(0).to(device)
    predicted_return = model(latest_data).cpu().item()


    return predicted_return


def load_and_preprocess_data(n_tickers):
    fin_data = pd.read_csv('fin_data.csv', low_memory=False)  # reads CSV file
    unique_tickers = fin_data['symbol'].unique()  # defines unique tickers the ticker symbols
    viable_tickers = []  # creates an empty set of valid tickers to be later appended to
    # Relevant fields for determining the value of a stock
    required_fields = ['symbol', 'forwardPE', 'trailingPE', 'bookValue', 'freeCashflow', 'debtToEquity',
                       'returnOnAssets', 'returnOnEquity', 'beta', 'marketCap', 'dividendYield', 'priceToBook']

    # While loop to make sure my valid_tickers is greater than or equal to n tickers
    while len(viable_tickers) < n_tickers:
        random_tickers = np.random.choice(unique_tickers, size=min(n_tickers * 2, len(unique_tickers)), replace=False)
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
    viable_tickers = [x for x in viable_tickers if not (isinstance(x, float) and math.isnan(x))]  # removes nan values

    # Creates a data frame of all the stock symbols and the required fundamentals
    stock_fundamentals_df = fin_data[fin_data['symbol'].isin(viable_tickers)][required_fields]

    # Downloads adjusted close price to calculate excess returns of each viable_ticker
    adj_close_df = pd.DataFrame(yf.download(viable_tickers, start=start_date, end=end_date)['Adj Close'])

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
    cumprod_df = adj_close_df.iloc[:, 1:].fillna(method='ffill')

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

    return stock_fundamentals_df, merged_df, adj_close_df


def select_top_stocks(portfolio_size):
    while True:
        stock_fundamentals_df, merged_df, adj_close_df = load_and_preprocess_data(max(portfolio_size * 2, 50))

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
        model = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.0001, max_iter=50000, random_state=42)
        model.fit(X_scaled, y)

        # Predict future returns
        merged_df.loc[X.index, 'predicted_return'] = model.predict(X_scaled)

        # Select top stocks
        top_stocks = merged_df.groupby('ticker')['predicted_return'].mean().nlargest(portfolio_size).index.tolist()

        if len(top_stocks) == portfolio_size:
            return top_stocks
        else:
            print(f"Found {len(top_stocks)} stocks. Retrying to match requested portfolio size of {portfolio_size}.")


def boy(tick, start, end, look):
    returns = []
    for t in tick:
        item = expectations(t, start, end, look)
        returns.append(item)
    return returns


def expected_return(weights, computed_returns):
    portfolio_return = np.dot(computed_returns, weights)
    return portfolio_return


#  Calculate standard deviation (volatility)

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)  # You don't need abs() for standard deviation, as variance is non-negative


# Sharpe Ratio: (Expected Return - Risk-Free Rate) / Volatility
def sharpe_ratio(weights, cov_matrix, risk_free_rate, precomputed_returns):
    port_return = expected_return(weights, precomputed_returns)
    port_volatility = standard_deviation(weights, cov_matrix)
    return (port_return - risk_free_rate) / port_volatility


# Negative Sharpe Ratio: Minimizing negative Sharpe ratio is equivalent to maximizing Sharpe ratio
def neg_sharpe_ratio(weights, cov_matrix, risk_free_rate, precomputed_returns):
    return -sharpe_ratio(weights, cov_matrix, risk_free_rate, precomputed_returns)


def optimize_portfolio(valid_tickers, adj_close_df, risk_free_rate, start_date, end_date):
    precomputed_returns = boy(valid_tickers, start_date, end_date, 30)
    print(precomputed_returns)
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()
    cov_matrix = log_returns.cov() * 252  # Annualize covariance matrix (252 trading days)

    # Constraints: weights must sum to 1 (fully invested portfolio)
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

    # Bounds for each weight: between 0 and 0.6 (for each ticker)
    bounds = [(0, 0.9) for _ in range(len(valid_tickers))]

    # Initial guess for weights (equally distributed)
    initial_weights = np.array([1 / len(valid_tickers)] * len(valid_tickers))

    # Minimize the negative Sharpe ratio
    optimized_results = minimize(neg_sharpe_ratio, initial_weights,
                                 args=(cov_matrix, risk_free_rate, precomputed_returns),
                                 method='SLSQP', constraints=constraints, bounds=bounds)

    # Extract optimal weights
    optimal_weights = optimized_results.x

    # Calculate the optimal portfolio's return and volatility

    optimal_portfolio_return = expected_return(precomputed_returns, optimal_weights)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
    optimal_sharpe_ratio = sharpe_ratio(optimal_weights, cov_matrix, risk_free_rate, precomputed_returns)
    return optimal_weights, optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio


iteration_portfolio = 0
best_sharpe_ratio = -np.inf
best_portfolio = None

while iteration_portfolio < max_iterations_portfolio:
    iteration_portfolio += 1
    print(f"\nIteration {iteration_portfolio} of {max_iterations_portfolio}")

    optimized_stock = set()
    while len(optimized_stock) < portfolio_size:
        new_optimized_stock = select_top_stocks(portfolio_size)
        optimized_stock.update(new_optimized_stock)
    sorted_optimized_stock = sorted(optimized_stock)
    print(f"Optimized stocks include: {sorted_optimized_stock}")

    adj_close_df = pd.DataFrame()
    valid_tickers = []

    for ticker in sorted_optimized_stock:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty and 'Adj Close' in data.columns and not data['Adj Close'].isnull().all():
                adj_close_df[ticker] = data['Adj Close']
                valid_tickers.append(ticker)
            else:
                print(f"Data for {ticker} is invalid, skipping...")
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error downloading data for {ticker}: {e}. Skipping...")

    print(f"Valid Stocks include {valid_tickers}")
    print(f"Portfolio size {len(valid_tickers)}")

    if adj_close_df.empty:
        print("No valid stock data available for analysis. Retrying with a new set of stocks.")
        continue

    pw, pr, pv, sr = optimize_portfolio(valid_tickers, adj_close_df, risk_free_rate, start_date, end_date)
    print(f"Expected Annual Return: {pr * 100:.4f}%")
    print(f"Expected Volatility: {pv * 100:.4f}%")
    print(f"Sharpe Ratio: {sr:.4f}")

    if sr > best_sharpe_ratio:
        best_sharpe_ratio = sr
        best_portfolio = (valid_tickers, pw, pr, pv, sr)

    if (sr >= target_sharpe_ratio and pr <= 1
            and pr >= target_ear and pv <= target_volatility):
        print("Target Sharpe Ratio and Expected Annual Return met!")
        break
    else:
        print("Target not met. Retrying with a new set of stocks.")

if iteration_portfolio == max_iterations_portfolio:
    print("\nMaximum iterations reached. Using the best portfolio found:")
    valid_tickers, optimal_weights, optimal_portfolio_return, optimal_portfolio_volatility = best_portfolio
    print(f"Best Sharpe Ratio: {best_sharpe_ratio:.4f}")
    print(f"Best Expected Annual Return: {optimal_portfolio_return * 100:.4f}%")
    print(f"Best Expected Volatility: {optimal_portfolio_volatility * 100:.4f}%")
else:
    print("\nOptimal portfolio found:")

# Code to delete stocks that have basically 0 weight

new_valid_tickers = []
new_optimal_weights0 = []

for i in range(len(pw)):
    if pw[i] >= 0.0001:
        new_valid_tickers.append(valid_tickers[i])
        new_optimal_weights0.append(pw[i])

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

N = float(input("Enter your initial backtest investment amount:"))


def capital_allocation():
    print("Capital Allocation for Initial Backtest Investment Amount")
    for ticker, weight in zip(new_valid_tickers, new_optimal_weights0):
        print(f"{ticker}: {round(N * weight, 2)} : Company Name: {yf.Ticker(ticker).info['longName']}, # of Shares :"
              f"{round(float(round(N * weight, 2)/float(yf.Ticker(ticker).info['currentPrice'])), 2)}")
    return capital_allocation


capital_allocation = capital_allocation()

fred = Fred(api_key='c496761920d5eedc780bf493ac3ae0e9')
ten_year_treasury_note = fred.get_series_latest_release('GS10') / 100
risk_free_rate = ten_year_treasury_note.iloc[-1]
rate = round(risk_free_rate * 100, 4)

data = yf.download(new_valid_tickers, start=start_date2, end=end_date2)['Adj Close']
years = (datetime.strptime(end_date2, '%Y-%m-%d') - datetime.strptime(start_date2, '%Y-%m-%d')).days / 365.25
returns = data.pct_change()
portfolio_returns = returns.dot(new_optimal_weights)
cumulative_returns = N * (1 + portfolio_returns).cumprod()
total_return = cumulative_returns.iloc[-1]
annualized_return = ((total_return / N) - 1) ** 1 / years
annualized_volatility = portfolio_returns.std() * np.sqrt(252)  # Assuming 252 trading days in a year
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

print(f"BACKTEST RESULTS")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_volatility:.2%}")
print(f"Total returns: ${round(total_return, 2)}")
print(f"Sharpe_ratio: {sharpe_ratio}")

# Sell strategy
# Strategy: Sell when the portfolio value reaches the predicted expected return
predicted_value = N * (1 + pr)

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

if total_return - N >= 0:
    print(f"Gain of ${round(total_return - N, 2)} on ${N} investment")
else:
    print(f"Loss of ${round(N - total_return, 2)} on ${N} investment")

plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns.index, cumulative_returns.values)
plt.title('Portfolio Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(True)
plt.show()

# Download data for the train data
data_train = yf.download(new_valid_tickers, start=datetime.today() - timedelta(days=3 * 365),
                         end=datetime.today() - timedelta(days=365))['Adj Close']
returns_train = data_train.pct_change(fill_method=None).dropna()
portfolio_returns_train = returns_train.dot(new_optimal_weights)
cumulative_returns_train = (1 + portfolio_returns_train).cumprod()

data_test = yf.download(new_valid_tickers, start=datetime.today() - timedelta(days=365),
                        end=end_date2)['Adj Close']
returns_test = data_test.pct_change(fill_method=None).dropna()
portfolio_returns_test = returns_test.dot(new_optimal_weights)
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


# Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x).squeeze(1)


# Initialize and train the model
model = LinearRegressionModel(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.03)
num_loops = 3000

for loop in range(num_loops):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Loops[{loop + 1}/{num_loops}], Loss: {loss.item():.2f}')

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
