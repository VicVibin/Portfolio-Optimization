#  Stock Model 11
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
from torch.cuda.amp import autocast, GradScaler


device = torch.device('cuda')
# Get the Risk Free Rate
fred = Fred(api_key='c496761920d5eedc780bf493ac3ae0e9')
ten_year_treasury_note = fred.get_series_latest_release('GS10') / 100
risk_free_rate = ten_year_treasury_note.iloc[-1]

#  Input parameters
t = int(input("Enter your initial portfolio optimizer end year:"))
t_1 = int(input("Enter the length of time data for optimization:"))
end_date = datetime.today() - timedelta(days=t * 365)
start_date = end_date - timedelta(days=t_1 * 365)

# Backtest dates
start_date2 = end_date.strftime('%Y-%m-%d')
end_date2 = (end_date + timedelta(days=365)).strftime('%Y-%m-%d')


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# PyTorch LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, forecast_steps):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True, num_layers=1)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()
        self.lstm2 = nn.LSTM(64, 64, batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(64, forecast_steps)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x = self.tanh(self.dense1(x[:, -1, :]))
        x = x.unsqueeze(1)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.output(x.squeeze(1))
        return x


class Conv1DModel(nn.Module):
    def __init__(self, input_size, forecast_steps):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(64 * 58, 64)
        self.output = nn.Linear(64, forecast_steps)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.tanh(self.conv2(x))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.output(x)
        return x


def calculate_macd(log_prices, short_period=12, long_period=26, signal_period=9):
    """Calculate MACD technical indicator"""
    short_ema = log_prices.ewm(span=short_period).mean()
    long_ema = log_prices.ewm(span=long_period).mean()
    macd = short_ema - long_ema
    return macd


def calculate_rsi(prices, periods=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def create_sequences(db_feat, db_resp, n_steps, forecast_steps):
    X, y = [], []
    for i in range(n_steps, len(db_feat) - forecast_steps + 1):
        X.append(db_feat[i - n_steps:i])
        y.append(db_resp[i:i + forecast_steps])
    X, y = np.array(X), np.array(y)
    return torch.FloatTensor(X), torch.FloatTensor(y)


def expectations(ticker, start, end, model, lookback, forecast_steps, epochs=150):
    print(f'Ticker being analyzed : "{ticker}')
    data = yf.download(ticker, start=start, end=end)
    data = pd.DataFrame(data)
    train_dates = pd.to_datetime(data.index)
    data['Date'] = train_dates
    data['Log_Return'] = np.log(data['Adj Close']).pct_change()
    data['Log_Price'] = np.log(data['Adj Close'])
    # Moving averages of log prices
    data['Log_MA5'] = data['Log_Price'].rolling(window=5).mean()
    data['Log_MA20'] = data['Log_Price'].rolling(window=20).mean()
    data['Log_MA50'] = data['Log_Price'].rolling(window=50).mean()
    # Volatility measures (using log returns)
    data['Volatility_5'] = data['Log_Return'].rolling(window=5).std()
    data['Volatility_20'] = data['Log_Return'].rolling(window=20).std()
    # Price momentum indicators
    data['RSI'] = calculate_rsi(data['Adj Close'], periods=14)
    data['Log_RSI'] = np.log(data['RSI'] + 1)  # Add 1 to handle zero values
    data['Log_Momentum'] = (1 + data['Log_Return']).cumprod()
    # Moving average convergence/divergence
    data['MACD'] = calculate_macd(data['Log_Return'])
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    features = ['Log_Price', 'Log_MA5', 'Log_MA20', 'Log_MA50', 'Volatility_5',
                'Volatility_20', 'Log_RSI', 'Log_Momentum', 'MACD']
    data[features] = StandardScaler().fit_transform(data[features])
    database = data[features]
    X = database.values
    y = data['Log_Return']

    X_scaled = StandardScaler().fit_transform(X)
    X_seq, y_seq = create_sequences(X_scaled, y, lookback, forecast_steps)
    train_dataset = StockDataset(X_seq, y_seq)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    input_size = X.shape[1]
    model = model(input_size, forecast_steps).to(device)
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        train_losses = []

        # Training Phase
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            # Enable mixed precision training with autocast
            with autocast():  # Automatically uses FP16 for eligible operations
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

            # Scales the loss and performs the backward pass
            scaler.scale(loss).backward()

            # Updates the model parameters with the scaled gradients
            scaler.step(optimizer)

            # Updates the scaler for the next iteration
            scaler.update()

            # Store the loss for this batch
            train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        predicted_returns = pd.Series(model(X_seq[-1:].shape[1]).to(device).cpu().numpy())
    historic_data = y[-252:]
    return predicted_returns, historic_data


def actual_return(ticker, end):
    actual_data = np.log(yf.download(ticker, end, end + timedelta(days=365))['Adj Close']).pct_change()
    actual_data = pd.Series(actual_data)
    return actual_data


def graph(historic, predicted, actual):
    historical_x = list(range(len(historic)))
    predicted_x = list(range(len(historical_x), len(historical_x + len(predicted))))
    plt.figure(figsize=(15, 8))
    plt.plot(historical_x, historic.values, color="green")
    plt.plot(predicted_x, actual.values, color="Blue")
    plt.plot(predicted_x, predicted.values, color="Red")
    plt.xlabel("Time Step")
    plt.ylabel("Log Return")
    plt.legend()
    plt.show()
    return plt.gcf()

def graph_function(ticker, start, end, end_2, model, lookback, forecast_steps):
    predicted_returns, historic_returns = expectations(ticker, start, end, model,lookback, forecast_steps)
    actual = actual_return(ticker, end_2)
    graph(historic_returns, predicted_returns, actual)
    return True


model = Conv1DModel
graph_function("AMZN", start_date, end_date, end_date2, model, lookback=60, forecast_steps=60)




