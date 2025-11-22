import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor 
from sklearn.pipeline import Pipeline

# ==========================================
# 1. LOAD OUTDOOR DATA (Hong Kong Scenario)
# ==========================================
try:
    df = pd.read_csv('outdoor_ray_tracing_data.csv')
    print(f"Outdoor Data Loaded: {len(df)} samples.")
except FileNotFoundError:
    print("Generating dummy OUTDOOR data for demonstration...")
    np.random.seed(42)
    df = pd.DataFrame({
        'Latitude': np.random.uniform(22.28, 22.29, 1000),
        'Longitude': np.random.uniform(114.14, 114.15, 1000),
        'PathLoss': np.random.uniform(80, 120, 1000) 
    })

X = df[['Latitude', 'Longitude']].values
y = df['PathLoss'].values

# ==========================================
# 2. DEFINE MODELS
# ==========================================

# Model 1: XGBoost
model_xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)

# Model 2: Neural Network (Requires Scaling for Lat/Lon)
model_nn = Pipeline([
    ('scaler', StandardScaler()), 
    ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', max_iter=5000, random_state=42))
])

# ==========================================
# 3. 5-FOLD CROSS VALIDATION LOOP
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*50)
print("OUTDOOR 5G MODEL VALIDATION REPORT")
print("="*50)

# Lists to store results per fold
xgb_r2, xgb_rmse = [], []
nn_r2, nn_rmse = [], []

fold = 1
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train & Test XGBoost
    model_xgb.fit(X_train, y_train)
    pred_xgb = model_xgb.predict(X_test)
    xgb_r2.append(r2_score(y_test, pred_xgb) * 100)
    xgb_rmse.append(np.sqrt(mean_squared_error(y_test, pred_xgb)))
    
    # Train & Test Neural Net
    model_nn.fit(X_train, y_train)
    pred_nn = model_nn.predict(X_test)
    nn_r2.append(r2_score(y_test, pred_nn) * 100)
    nn_rmse.append(np.sqrt(mean_squared_error(y_test, pred_nn)))
    
    print(f"Fold {fold}: XGB Acc={xgb_r2[-1]:.2f}% | NN Acc={nn_r2[-1]:.2f}%")
    fold += 1

# Calculate Averages
avg_xgb_acc = np.mean(xgb_r2)
avg_xgb_rmse = np.mean(xgb_rmse)
avg_nn_acc = np.mean(nn_r2)
avg_nn_rmse = np.mean(nn_rmse)

print("-" * 50)
print(f"FINAL RESULTS (Average of 5 Folds):")
print(f"XGBoost Accuracy:     {avg_xgb_acc:.2f}%  (RMSE: {avg_xgb_rmse:.3f} dB)")
print(f"Neural Net Accuracy:  {avg_nn_acc:.2f}%  (RMSE: {avg_nn_rmse:.3f} dB)")
print("=" * 50)

# ==========================================
# 4. GENERATE FINAL PLOTS
# ==========================================
# Create a standard split for the visual graphs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_test)

model_nn.fit(X_train, y_train)
pred_nn = model_nn.predict(X_test)

plt.figure(figsize=(14, 6))

# --- PLOT 1: SCATTER PLOT (Accuracy Visualization) ---
plt.subplot(1, 2, 1)
plt.scatter(y_test, pred_xgb, alpha=0.6, c='blue', s=20, label=f'XGBoost ({avg_xgb_acc:.1f}%)')
plt.scatter(y_test, pred_nn, alpha=0.6, c='green', s=20, label=f'Neural Net ({avg_nn_acc:.1f}%)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.title(f"Outdoor Accuracy: Measured vs Predicted\n(Hong Kong 5G Scenario)", fontweight='bold')
plt.xlabel("True Path Loss (Ray Tracing)")
plt.ylabel("Predicted Path Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# --- PLOT 2: BER vs SNR (System Performance) ---
def ldpc_ber(snr_db):
    # Outdoor threshold is often lower (e.g., 25dB) due to different coding rates
    return 0.5 / (1 + np.exp(2.0 * (snr_db - 25.5)))

def uncoded_ber(snr_db):
    return 0.2 / (10**(snr_db/10) * 0.05)

ebno = np.arange(20, 35, 0.1)

# Shift the curve by the average RMSE error
ber_uncoded = [uncoded_ber(e) for e in ebno]
ber_true = [ldpc_ber(e) for e in ebno]
ber_xgb = [ldpc_ber(e - avg_xgb_rmse) for e in ebno]
ber_nn = [ldpc_ber(e - avg_nn_rmse) for e in ebno]

plt.subplot(1, 2, 2)
plt.semilogy(ebno, ber_uncoded, 'k--', alpha=0.4, label='Theoretical Uncoded')
plt.semilogy(ebno, ber_true, 'r-o', lw=2, label='Original CDL Model')
plt.semilogy(ebno, ber_xgb, 'b-', lw=2, label=f'XGBoost ({avg_xgb_acc:.1f}% Acc)')
plt.semilogy(ebno, ber_nn, 'g-.', lw=2, label=f'Neural Net ({avg_nn_acc:.1f}% Acc)')

plt.title(f"Outdoor 5G System Performance\n(CDL Channel Customization)", fontweight='bold')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.ylim(1e-5, 1)
plt.legend()
plt.grid(True, which="both", alpha=0.3)

# Add Stats Box
stats_text = (f"PERFORMANCE METRICS:\n"
              f"--------------------\n"
              f"XGB Acc: {avg_xgb_acc:.2f}%\n"
              f"NN Acc:  {avg_nn_acc:.2f}%")
plt.text(20.5, 0.0001, stats_text, fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))

plt.tight_layout()
plt.show()