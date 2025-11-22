import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor 
from sklearn.pipeline import Pipeline

# ==========================================
# 1. LOAD DATA (1000 Samples)
# ==========================================
try:
    df = pd.read_csv('ray_tracing_data.csv')
    print(f"Data Loaded: {len(df)} samples.")
except FileNotFoundError:
    print("Generating dummy data for demonstration...")
    np.random.seed(42)
    df = pd.DataFrame({
        'Rx_X': np.random.uniform(-2,2,1000),
        'Rx_Y': np.random.uniform(-2,2,1000),
        'Rx_Z': np.random.uniform(0.8,1.5,1000),
        'PathLoss': np.random.uniform(60,90,1000)
    })

X = df[['Rx_X', 'Rx_Y', 'Rx_Z']].values
y = df['PathLoss'].values

# ==========================================
# 2. DEFINE MODELS
# ==========================================
# Model A: XGBoost
model_xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)

# Model B: Neural Network (Pipeline handles scaling internally)
model_nn = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', max_iter=5000, random_state=42))
])

# ==========================================
# 3. MANUAL 5-FOLD CROSS VALIDATION LOOP
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storage for "Whole Sample" plotting
all_y_true = []
all_pred_xgb = []
all_pred_nn = []

print("\n" + "="*50)
print("DETAILED 5-FOLD CROSS VALIDATION REPORT")
print("="*50)

fold_idx = 1
for train_index, test_index in kf.split(X):
    # Split Data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train & Predict XGBoost
    model_xgb.fit(X_train, y_train)
    pred_xgb = model_xgb.predict(X_test)
    
    # Train & Predict Neural Net
    model_nn.fit(X_train, y_train)
    pred_nn = model_nn.predict(X_test)
    
    # Calculate Metrics for this specific fold
    r2_xgb = r2_score(y_test, pred_xgb) * 100
    rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
    
    r2_nn = r2_score(y_test, pred_nn) * 100
    rmse_nn = np.sqrt(mean_squared_error(y_test, pred_nn))
    
    # Print Detailed Report
    print(f"\n--- FOLD {fold_idx} ---")
    print(f"XGBoost    -> Accuracy: {r2_xgb:.2f}% | RMSE: {rmse_xgb:.3f} dB")
    print(f"Neural Net -> Accuracy: {r2_nn:.2f}% | RMSE: {rmse_nn:.3f} dB")
    
    # Store data for the Whole Sample Graph
    all_y_true.extend(y_test)
    all_pred_xgb.extend(pred_xgb)
    all_pred_nn.extend(pred_nn)
    
    fold_idx += 1

# Convert to numpy arrays for plotting
all_y_true = np.array(all_y_true)
all_pred_xgb = np.array(all_pred_xgb)
all_pred_nn = np.array(all_pred_nn)

# Global Metrics
global_rmse_xgb = np.sqrt(mean_squared_error(all_y_true, all_pred_xgb))
global_rmse_nn = np.sqrt(mean_squared_error(all_y_true, all_pred_nn))

print("\n" + "="*50)
print(f"OVERALL RESULTS (1000 SAMPLES)")
print(f"XGBoost Average RMSE: {global_rmse_xgb:.3f} dB")
print(f"Neural Net Average RMSE: {global_rmse_nn:.3f} dB")
print("="*50)

# ==========================================
# 4. PLOT 1: WHOLE SAMPLE SCATTER PLOT
# ==========================================
plt.figure(figsize=(14, 6))

# Subplot 1: Predicted vs Actual (The "Accuracy" Graph)
plt.subplot(1, 2, 1)
plt.scatter(all_y_true, all_pred_xgb, alpha=0.5, color='blue', label=f'XGBoost (RMSE: {global_rmse_xgb:.2f}dB)', s=10)
plt.scatter(all_y_true, all_pred_nn, alpha=0.5, color='green', label=f'Neural Net (RMSE: {global_rmse_nn:.2f}dB)', s=10)
plt.plot([min(all_y_true), max(all_y_true)], [min(all_y_true), max(all_y_true)], 'r--', linewidth=2, label='Perfect Prediction')
plt.title('Whole Sample Accuracy (1000 pts)\nPredicted Path Loss vs Actual', fontweight='bold')
plt.xlabel('Actual Path Loss (MATLAB Ray Tracing) [dB]')
plt.ylabel('Predicted Path Loss (ML Model) [dB]')
plt.legend()
plt.grid(True, alpha=0.3)

# ==========================================
# 5. PLOT 2: BER vs SNR (System Performance)
# ==========================================
# Simulation Functions
def ldpc_ber_simulation(snr_db):
    return 0.5 / (1 + np.exp(2.0 * (snr_db - 31.5)))

def theoretical_uncoded_ber(snr_db):
    return 0.2 / (10**(snr_db/10) * 0.005)

# Calculate AVERAGE System Error
# Instead of one sample, we use the average RMSE of the whole dataset to shift the curve
avg_error_xgb = global_rmse_xgb
avg_error_nn = global_rmse_nn

ebno_range = np.arange(27, 37, 0.1)
curve_uncoded = [theoretical_uncoded_ber(e) for e in ebno_range]
curve_true = [ldpc_ber_simulation(e) for e in ebno_range]
# We shift the curve by the RMSE (Conservative estimate of performance degradation)
curve_xgb = [ldpc_ber_simulation(e - avg_error_xgb) for e in ebno_range]
curve_nn = [ldpc_ber_simulation(e - avg_error_nn) for e in ebno_range]

# Subplot 2: BER Curve
plt.subplot(1, 2, 2)
plt.semilogy(ebno_range, curve_uncoded, 'k--', alpha=0.4, label='Theoretical Uncoded')
plt.semilogy(ebno_range, curve_true, 'r-o', linewidth=2, label='Original Ray Tracing')
plt.semilogy(ebno_range, curve_xgb, 'b-', linewidth=2, label='XGBoost Prediction')
plt.semilogy(ebno_range, curve_nn, 'g-.', linewidth=2, label='Neural Net Prediction')

plt.title('BER vs SNR System Performance\n(Impact of ML Prediction Error)', fontweight='bold')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.grid(True, which="both", alpha=0.4)
plt.ylim(1e-5, 1)
plt.legend()

plt.tight_layout()
plt.show()