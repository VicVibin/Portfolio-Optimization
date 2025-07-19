import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta


def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data


def engineer_features(data):
    """
    Create and transform features for the model using log transformations
    where appropriate
    """
    # Price-based features
    data['Log_Price'] = np.log(data['Adj Close'])

    # Returns
    data['Log_Return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))

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

    # Moving average convergence/divergence
    data['MACD'] = calculate_macd(data['Log_Price'])

    # Relative volume

    return data


def calculate_rsi(prices, periods=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(log_prices, short_period=12, long_period=26, signal_period=9):
    """Calculate MACD technical indicator"""
    short_ema = log_prices.ewm(span=short_period).mean()
    long_ema = log_prices.ewm(span=long_period).mean()
    macd = short_ema - long_ema
    return macd


def prepare_data(data, time_step=60, forecast_days=252):
    """
    Prepare data for LSTM model with multiple features
    """
    # Select features for the model
    feature_columns = [
        'Log_Price', 'Log_Return',
        'Log_MA5', 'Log_MA20', 'Log_MA50',
        'Volatility_5', 'Volatility_20',
        'Log_RSI', 'MACD'
    ]

    # Drop any rows with NaN values
    data = data.dropna()

    # Scale all features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[feature_columns])
    scaled_data = pd.DataFrame(scaled_data, columns=feature_columns, index=data.index)

    # Prepare sequences for LSTM
    X, y = [], []
    for i in range(time_step, len(scaled_data) - forecast_days):
        # Input sequence: all features
        X.append(scaled_data.iloc[i - time_step:i].values)
        # Target: next forecast_days log returns
        y.append(data['Log_Return'].iloc[i:i + forecast_days].values)

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler, feature_columns


def build_model(X, forecast_days=252):
    """
    Build an enhanced LSTM model with dropout for regularization
    """
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(forecast_days)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_future(model, data, scaler, feature_columns, time_step=60, forecast_days=252):
    """
    Predict future returns using the trained model
    """
    # Prepare the last time_step days of data
    last_sequence = data[feature_columns].iloc[-time_step:].values
    # Scale the sequence
    last_sequence_scaled = scaler.transform(last_sequence)
    # Reshape for LSTM input
    last_sequence_scaled = last_sequence_scaled.reshape(1, time_step, len(feature_columns))

    # Predict future returns
    predicted_returns = model.predict(last_sequence_scaled)

    # Convert predictions to actual values
    initial_price = data['Adj Close'].iloc[-1]
    cumulative_returns = np.cumsum(predicted_returns.flatten())
    predicted_prices = initial_price * np.exp(cumulative_returns)

    return predicted_prices


# Example usage
def run_prediction(ticker, start_date, end_date):
    # Download and prepare data
    data = download_data(ticker, start_date, end_date)
    data = engineer_features(data)

    # Prepare data for LSTM
    time_step = 60
    forecast_days = 252
    X, y, scaler, feature_columns = prepare_data(data, time_step, forecast_days)

    # Build and train model
    model = build_model(X, forecast_days)
    model.fit(X, y, epochs=1000, batch_size=32, validation_split=0.25, verbose=1)

    # Make predictions
    predicted_prices = predict_future(model, data, scaler, feature_columns, time_step, forecast_days)

    # Calculate expected return
    expected_return = predicted_prices[-1] / data['Adj Close'].iloc[-1] - 1

    return predicted_prices, expected_return, data

def plot_price_comparison(historical_data, predicted_prices, ticker, actual_prices):
    # Create future dates for predictions
    last_date = historical_data.index[-1]
    future_dates = pd.date_range(last_date, periods=len(predicted_prices) + 1, freq='B')[1:]

    # Create a DataFrame for predicted prices
    predicted_df = pd.Series(predicted_prices, index=future_dates)
    matched_adj_close = actual_prices['Adj Close']

    # Get the last 252 days (1 year) of historical data for better visualization
    historical_subset = historical_data['Adj Close'].last('252D')

    # Create the visualization
    plt.figure(figsize=(15, 8))

    # Plot historical prices
    plt.plot(historical_subset.index, historical_subset.values,
             label='Historical Prices', color='blue', linewidth=2)

    # Plot predicted prices
    plt.plot(predicted_df.index, predicted_df.values,
             label='Predicted Prices', color='red')
    plt.plot(matched_adj_close.index, matched_adj_close.values, label='Actual Prices', color='green')

    # Add intersection point
    plt.scatter(last_date, historical_subset[-1],
                color='green', s=100, zorder=5,
                label='Intersection Point')

    # Customize the plot
    plt.title(f'{ticker} Stock Price Prediction', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add text box with metrics
    returns = (predicted_prices[-1] / historical_subset[-1] - 1) * 100
    volatility = np.std(np.diff(np.log(predicted_prices))) * np.sqrt(252) * 100

    textstr = f'Predicted Return: {returns:.2f}%\nPredicted Volatility: {volatility:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=props)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return plt.gcf()


# Example usage
def visualize_predictions(ticker, start_date, end_date, actual_prices):
    # Get the predictions using the previous code
    predicted_prices, expected_return, historical_data = run_prediction(ticker, start_date, end_date)

    # Create and show the visualization
    fig = plot_price_comparison(historical_data, predicted_prices, ticker, actual_prices)
    plt.show()

    return fig


# Run the visualization
ticker = 'MSFT'
end_date = datetime.today() - timedelta(days=365)
start_date = end_date - timedelta(days=5*365)

actual_prices = yf.download(ticker, end_date, end_date + timedelta(days=365))

fig = visualize_predictions(ticker, start_date, end_date, actual_prices)


