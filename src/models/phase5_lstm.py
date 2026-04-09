import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONSTANTS & HYPERPARAMETERS
# ---------------------------------------------------------
SEQ_LENGTH = 7
EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 0.001
HIDDEN_DIM = 64
NUM_LAYERS = 2

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_weather_all_features.csv')

print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# Drop date, keep 'load' as target (index 0 for easy tracking) 
features = ['load', 'temp_max', 'temp_min', 'temp_avg', 'temp_range', 
            'precipitation', 'wind_speed', 'day_of_week', 'day_of_month', 
            'month', 'is_weekend', 'month_sin', 'month_cos', 'day_sin', 
            'day_cos', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_std_7']

data = df[features].values

# ---------------------------------------------------------
# 2. SCALE DATA
# ---------------------------------------------------------
# We fit the scaler on the first 80% to avoid data leakage
train_size = int(len(data) * 0.8)
train_data = data[:train_size, :]
test_data = data[train_size:, :]

scaler = MinMaxScaler()
# Fit on train, transform both
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# ---------------------------------------------------------
# 3. CREATE SEQUENCES
# ---------------------------------------------------------
def create_sequences(data_seq, seq_length):
    xs = []
    ys = []
    for i in range(len(data_seq) - seq_length):
        x = data_seq[i:(i + seq_length)]
        y = data_seq[i + seq_length, 0] # load is index 0
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

# Convert to PyTorch Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ---------------------------------------------------------
# 4. DATA LOADERS
# ---------------------------------------------------------
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------------------------------------
# 5. DEFINE LSTM MODEL
# ---------------------------------------------------------
class LoadLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LoadLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

INPUT_DIM = X_train.shape[2]
OUTPUT_DIM = 1

model = LoadLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------------------------------------------
# 6. TRAINING LOOP
# ---------------------------------------------------------
print("\nTraining LSTM Network...")
model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {epoch_loss/len(train_loader):.4f}")

# ---------------------------------------------------------
# 7. EVALUATION
# ---------------------------------------------------------
print("\nEvaluating Model on Test Data...")
model.eval()
with torch.no_grad():
    test_preds = model(X_test)

# Inverse transform to get actual MW load values
# (We need to construct a dummy array for the scaler to inverse transform properly since it expects features count)
dummy_pred = np.zeros((len(test_preds), INPUT_DIM))
dummy_pred[:, 0] = test_preds.squeeze().numpy()
inv_preds = scaler.inverse_transform(dummy_pred)[:, 0]

dummy_actual = np.zeros((len(y_test), INPUT_DIM))
dummy_actual[:, 0] = y_test.squeeze().numpy()
inv_actual = scaler.inverse_transform(dummy_actual)[:, 0]

mae = mean_absolute_error(inv_actual, inv_preds)
rmse = root_mean_squared_error(inv_actual, inv_preds)

print("\n===========================================================")
print("  PHASE 5 DEEP LEARNING (LSTM) RESULTS")
print("===========================================================")
print(f"Test MAE  : {mae:,.0f}")
print(f"Test RMSE : {rmse:,.0f}")
print("===========================================================")
print("\nNOTE: Phase 4 Random Forest achieved CV MAE: ~4,168")
