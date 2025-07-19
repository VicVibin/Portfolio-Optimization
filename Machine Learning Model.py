import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

features = ['Close', 'Returns', 'MA50', 'MA200', 'Volatility']
start_date = '2018-01-01'
end_date = '2024-01-01'

# Custom Dataset class for PyTorch
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


def prepare_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change()
    data['MA50'] = data['Adj Close'].rolling(window=50).mean()
    data['MA200'] = data['Adj Close'].rolling(window=200).mean()
    data['Volatility'] = data['Returns'].rolling(window=50).std()
    data.dropna(inplace=True)
    data['Target'] = data['Adj Close'].shift(-252) / data['Adj Close'] - 1
    print(data['Target'])
    data = data[:-252]
    print(data)
    return data


data = prepare_data("META")
print(data['Target'])
X = data[features].values
y = data['Target'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Create sequences
def create_sequences(X, y, look_back=30):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)])
        ys.append(y[i + look_back - 1])
    return np.array(Xs), np.array(ys)


look_back = 30
X_seq, y_seq = create_sequences(X_scaled, y, look_back)
print(f'"X_seq shape:{X_seq.shape}')
print(f"y_seq shape: {y_seq.shape}")

# Split data
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockLSTM(input_size=X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
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

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

# Load best model
model.load_state_dict(best_model_state)

# Save the model
torch.save(model.state_dict(), 'stock_lstm.pth')

# Make predictions
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_pred = model(X_test_tensor).cpu().numpy()

# Latest prediction
latest_data = torch.FloatTensor(X_scaled[-look_back:]).unsqueeze(0).to(device)
predicted_return = model(latest_data).cpu().item()
last_actual_return = y_test[-1]




print(f'\nReturn Predictions and Actuals:')
print(f'Predicted 1-Year Return: {predicted_return:.2%}')
print(f'Last Actual 1-Year Return: {last_actual_return:.2%}')

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_train[-252:])), y_train[-252:], label='Historic Returns', color='blue')
plt.plot(range(len(y_train[-252:]), len(y_train[-252:]) + len(y_test)),y_test, label='Actual Returns', color='green')
plt.plot(range(len(y_train[-252:]), len(y_train[-252:]) + len(y_pred)), y_pred, label='Predicted Returns', color='red')
plt.title('One-Year Returns: Actual vs Predicted')
plt.xlabel('Days')
plt.ylabel('Return (%)')
plt.legend()
plt.show()

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
