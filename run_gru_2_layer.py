import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Reshape, RepeatVector, TimeDistributed, LayerNormalization, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import joblib

# --- 1. CẤU HÌNH CÁC SIÊU THAM SỐ (Hyperparameters) ---
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
N_IN = 72
N_OUT = 24
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
UNITS = 192
EPOCHS = 150
BATCH_SIZE = 32
DATA_FILE = 'HoaBinh.csv'
SELECTED_FEATURES = [
    'Mực nước hồ (m)',
    'Lưu lượng đến hồ (m³/s)',
    'Tổng lưu lượng xả (m³/s)[Thực tế]'
]
TARGET_FEATURE = 'Mực nước hồ (m)'
NUM_FEATURES = len(SELECTED_FEATURES)
MODEL_NAME = 'gru_2_layer'

TREND_BASELINE_WINDOW = 24

print(f"--- BẮT ĐẦU CHẠY MÔ HÌNH: {MODEL_NAME} ---")

# --- 2. TẢI VÀ XỬ LÝ DỮ LIỆU ---
print("Đang tải dữ liệu...")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"LỖI: Không tìm thấy file {DATA_FILE}. Vui lòng kiểm tra lại.")

df = pd.read_csv(DATA_FILE, parse_dates=['thoi_gian'], dayfirst=True, na_values='--')
df = df.sort_values(by='thoi_gian')

# Tạo đặc trưng thời gian điều hoà (harmonic)
df['hour'] = df['thoi_gian'].dt.hour.astype(int)
df['dayofweek'] = df['thoi_gian'].dt.dayofweek.astype(int)
df['dayofyear'] = df['thoi_gian'].dt.dayofyear.astype(int)
df['hod_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hod_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

EXTRA_FEATURES = ['hod_sin','hod_cos','dow_sin','dow_cos','doy_sin','doy_cos']
ALL_FEATURES = SELECTED_FEATURES + EXTRA_FEATURES

df[ALL_FEATURES] = df[ALL_FEATURES].ffill()
data = df[ALL_FEATURES].values.astype(float)

# Cập nhật lại số lượng đặc trưng cho mô hình và inverse-transform
NUM_FEATURES = len(ALL_FEATURES)

# --- 3. PHÂN CHIA VÀ CHUẨN HÓA DỮ LIỆU ---
n = len(data)
n_test = int(n * TEST_SIZE)
n_val = int(n * VALIDATION_SIZE)
n_train = n - n_test - n_val
train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]
print(f"Tổng số mẫu: {n}, Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

scaler = StandardScaler()
scaler = scaler.fit(train_data)
joblib.dump(scaler, f'scaler_{MODEL_NAME}.pkl')
print(f"Đã lưu scaler vào file 'scaler_{MODEL_NAME}.pkl'")

train_scaled = scaler.transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# Ước lượng biên độ thay đổi tối đa của mục tiêu trên miền scaled để ràng buộc delta dự báo
target_train_scaled = train_scaled[:, 0]
_abs_deltas = np.abs(np.diff(target_train_scaled))
MAX_DELTA_SCALED = float(np.quantile(_abs_deltas, 0.90)) if _abs_deltas.size > 0 else 0.05
if MAX_DELTA_SCALED < 1e-4:
    MAX_DELTA_SCALED = 0.05

# --- 4. HÀM TẠO CỬA SỔ TRƯỢT (SLIDING WINDOW) ---
def create_sequences(data, n_in, n_out):
    X, y = [], []
    for i in range(len(data) - n_in - n_out + 1):
        X.append(data[i:(i + n_in), :])
        y.append(data[(i + n_in):(i + n_in + n_out), 0])
    return np.array(X), np.array(y)

def compute_trend_baseline_scaled(X_seq, n_out, w):
    w = min(w, X_seq.shape[1])
    tgt = X_seq[:, -w:, 0]
    t = np.arange(w, dtype=np.float32)
    t_c = t - t.mean()
    den = np.sum(t_c ** 2)
    num = np.sum((tgt - tgt.mean(axis=1, keepdims=True)) * t_c, axis=1, keepdims=True)
    slope = num / (den + 1e-8)
    q = np.quantile(np.abs(slope), 0.99)
    slope = np.clip(slope, -q, q)
    last = X_seq[:, -1, 0].reshape(-1, 1)
    steps = np.arange(1, n_out + 1, dtype=np.float32).reshape(1, n_out)
    base = last + slope * steps
    return base.reshape(-1, n_out, 1)

print("Đang tạo mẫu cửa sổ trượt (windowing)...")
X_train, y_train = create_sequences(train_scaled, N_IN, N_OUT)
X_val, y_val = create_sequences(val_scaled, N_IN, N_OUT)
X_test, y_test = create_sequences(test_scaled, N_IN, N_OUT)

# kiểm tra đủ window
if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise ValueError("Không đủ mẫu để tạo window. Giảm N_IN/N_OUT hoặc tăng kích thước tập dữ liệu.")

# reshape y cho phù hợp (n_samples, N_OUT, 1)
y_train = y_train.reshape(-1, N_OUT, 1)
y_val = y_val.reshape(-1, N_OUT, 1)
y_test = y_test.reshape(-1, N_OUT, 1)
train_baseline_scaled = np.repeat(X_train[:, -1, 0].reshape(-1, 1, 1), N_OUT, axis=1)
val_baseline_scaled = np.repeat(X_val[:, -1, 0].reshape(-1, 1, 1), N_OUT, axis=1)
test_baseline_scaled = np.repeat(X_test[:, -1, 0].reshape(-1, 1, 1), N_OUT, axis=1)
y_train = y_train - train_baseline_scaled
y_val = y_val - val_baseline_scaled
y_test = y_test - test_baseline_scaled

print(f"Shape của X_train: {X_train.shape}, Shape của y_train: {y_train.shape}")

# In thông tin shapes và thống kê target
print("SHAPES:")
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_val", X_val.shape)
print("y_val", y_val.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

print("TARGET raw stats (target column index 0):")
print("train min/max:", train_data[:,0].min(), train_data[:,0].max(), "mean:", train_data[:,0].mean())
print("val   min/max:", val_data[:,0].min(),   val_data[:,0].max(),   "mean:", val_data[:,0].mean())
print("test  min/max:", test_data[:,0].min(),  test_data[:,0].max(),  "mean:", test_data[:,0].mean())

# Trọng số theo từng bước (ưu tiên các bước gần hơn)
time_weights = tf.constant(np.ones((1, N_OUT, 1), dtype=np.float32))

# Huber + làm mượt theo thời gian
def make_weighted_huber_smooth_loss(time_w, delta=1.0, smooth_lambda=0.001):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        ae = tf.abs(e)
        huber = tf.where(ae <= delta, 0.5 * tf.square(e), delta * (ae - 0.5 * delta))
        huber = huber * time_w
        loss_main = tf.reduce_mean(huber)
        diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        smooth = tf.reduce_mean(tf.abs(diff))
        return loss_main + smooth_lambda * smooth
    return loss

weighted_loss_base = make_weighted_huber_smooth_loss(time_weights, delta=1.0, smooth_lambda=2e-4)
weighted_loss_err = make_weighted_huber_smooth_loss(time_weights, delta=1.0, smooth_lambda=5e-4)

# --- 5. XÂY DỰNG MÔ HÌNH GRU 2 LỚP ---
print(f"--- Đang xây dựng mô hình: {MODEL_NAME} ---")
model = Sequential()
model.add(Input(shape=(N_IN, NUM_FEATURES)))
model.add(GRU(UNITS, return_sequences=False, recurrent_dropout=0.05))
model.add(LayerNormalization())
model.add(Dropout(0.15))
model.add(RepeatVector(N_OUT))
model.add(GRU(UNITS, return_sequences=True, recurrent_dropout=0.05))
model.add(LayerNormalization())
model.add(Dropout(0.15))
model.add(TimeDistributed(Dense(96, activation='relu')))
model.add(Dropout(0.15))
model.add(TimeDistributed(Dense(1, activation='linear')))

# optimizer & loss
model.compile(optimizer=Adam(learning_rate=3e-4, clipnorm=1.0), loss=weighted_loss_base, metrics=['mae'])
model.summary()

# --- 6. HUẤN LUYỆN MÔ HÌNH ---
model_filename = f'model_{MODEL_NAME}.h5'
plot_loss_filename = f'bieu_do_loss_{MODEL_NAME}.png'
plot_pred_filename = f'ket_qua_du_bao_{MODEL_NAME}.png'
plot_error_step_filename = f'sai_so_tung_buoc_{MODEL_NAME}.png'

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1, min_lr=1e-7)

print(f"Đang huấn luyện mô hình {MODEL_NAME}...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    shuffle=False,
    verbose=1
)
model.save(model_filename)
print(f"Đã lưu mô hình vào file '{model_filename}'")

# Huấn luyện mô hình sửa sai số (residual)
print("Đang tạo nhãn sai số cho mô hình sửa sai số...")
y_train_base = model.predict(X_train)
y_val_base = model.predict(X_val)
res_train = y_train - y_train_base
res_val = y_val - y_val_base

# Tạo đặc trưng chỉ số bước thời gian cho decoder (0..1)
step_idx = np.linspace(0, 1, N_OUT, dtype=np.float32).reshape(1, N_OUT, 1)
err_train_in = np.concatenate([y_train_base, np.repeat(step_idx, y_train_base.shape[0], axis=0)], axis=2)
err_val_in = np.concatenate([y_val_base,   np.repeat(step_idx, y_val_base.shape[0],   axis=0)], axis=2)

print("--- Xây dựng mô hình sửa sai số (error_model) ---")
error_model = Sequential()
error_model.add(Input(shape=(N_OUT, 2)))
error_model.add(GRU(UNITS//2, return_sequences=True, recurrent_dropout=0.05))
error_model.add(LayerNormalization())
error_model.add(Dropout(0.15))
error_model.add(TimeDistributed(Dense(48, activation='relu')))
error_model.add(Dropout(0.15))
error_model.add(TimeDistributed(Dense(1, activation='linear')))
error_model.compile(optimizer=Adam(learning_rate=3e-4, clipnorm=1.0), loss=weighted_loss_err, metrics=['mae'])
err_early = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
err_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1, min_lr=1e-7)
print("Đang huấn luyện error_model...")
error_model.fit(
    err_train_in, res_train,
    epochs=max(30, EPOCHS//2),
    batch_size=BATCH_SIZE,
    validation_data=(err_val_in, res_val),
    callbacks=[err_early, err_reduce],
    shuffle=False,
    verbose=1
)
error_model.save(f'model_{MODEL_NAME}_err.h5')
print("Đã lưu error_model")

# --- 7. HÀM UN-SCALE CHÍNH XÁC CHO TARGET ---
def unscale_target_array(y_scaled_3d, scaler, target_col_index=0):
    """
    y_scaled_3d: shape (n_samples, N_OUT, 1) or (n_samples, N_OUT)
    scaler: MinMaxScaler đã fit trên NUM_FEATURES
    trả về shape (n_samples, N_OUT, 1) sau khi inverse transform
    """
    arr = y_scaled_3d.reshape(-1)  # (n_samples * N_OUT,)
    nsamples = y_scaled_3d.shape[0]
    nsteps = y_scaled_3d.shape[1]
    tmp = np.zeros((nsamples * nsteps, NUM_FEATURES))
    tmp[:, target_col_index] = arr
    inv = scaler.inverse_transform(tmp)
    return inv[:, target_col_index].reshape(nsamples, nsteps, 1)

# --- 8. ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ---
print(f"Đang đánh giá mô hình {MODEL_NAME} trên tập Test...")
y_base_test = model.predict(X_test)
err_test_in = np.concatenate([y_base_test, np.repeat(step_idx, y_base_test.shape[0], axis=0)], axis=2)
y_err_test = error_model.predict(err_test_in)
y_pred_scaled = y_base_test + y_err_test

# Unscale đúng cách
y_pred_scaled_abs = y_pred_scaled + test_baseline_scaled
y_test_scaled_abs = y_test + test_baseline_scaled
y_pred_unscaled = unscale_target_array(y_pred_scaled_abs, scaler, target_col_index=0)
y_test_unscaled = unscale_target_array(y_test_scaled_abs, scaler, target_col_index=0)

# Hiệu chỉnh tuyến tính dựa trên tập validation
val_base_pred = model.predict(X_val)
err_val_in_eval = np.concatenate([val_base_pred, np.repeat(step_idx, val_base_pred.shape[0], axis=0)], axis=2)
val_err_pred = error_model.predict(err_val_in_eval)
val_pred_scaled = val_base_pred + val_err_pred
val_pred_scaled_abs = val_pred_scaled + val_baseline_scaled
val_pred_unscaled = unscale_target_array(val_pred_scaled_abs, scaler, target_col_index=0)
val_true_unscaled = unscale_target_array(y_val + val_baseline_scaled, scaler, target_col_index=0)
# Fit tuyến tính per-horizon: y_true ≈ a[h]*y_pred + b[h]
a = np.ones((1, N_OUT, 1))
b = np.zeros((1, N_OUT, 1))

# Áp dụng hiệu chỉnh cho tập test và cắt về khoảng min-max quan sát trên train
target_min = train_data[:,0].min()
target_max = train_data[:,0].max()
y_pred_unscaled_cal = a * y_pred_unscaled + b

y_pred_for_eval = np.clip(y_pred_unscaled_cal, target_min, target_max)

# --- 8a. Đánh giá TỔNG QUAN ---
y_test_flat = y_test_unscaled.reshape(-1)
y_pred_flat = y_pred_for_eval.reshape(-1)
mae = mean_absolute_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
r2 = r2_score(y_test_flat, y_pred_flat)

print("\n--- KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN (Sau hiệu chỉnh) ---")
print(f"Mô hình: {MODEL_NAME}")
print(f"Chỉ số MAE: {mae:.6f} (mét)")
print(f"Chỉ số RMSE: {rmse:.6f} (mét)")
print(f"Chỉ số R²: {r2:.6f}")

# --- 8b. Sai số theo từng bước (horizon) ---
print("\n--- KẾT QUẢ SAI SỐ THEO TỪNG BƯỚC DỰ BÁO (HORIZON) ---")
mae_per_step = np.mean(np.abs(y_test_unscaled - y_pred_for_eval), axis=0).flatten()
rmse_per_step = np.sqrt(np.mean(np.square(y_test_unscaled - y_pred_for_eval), axis=0)).flatten()
for i, (mae_val, rmse_val) in enumerate(zip(mae_per_step, rmse_per_step)):
    print(f"  Giờ t+{i+1}: MAE={mae_val:.6f} | RMSE={rmse_val:.6f}")

# --- 9. Baseline persistence để so sánh ---
last_values = X_test[:, -1, 0]
baseline_pred_scaled = np.repeat(last_values.reshape(-1,1), N_OUT, axis=1).reshape(-1, N_OUT, 1)
baseline_unscaled = unscale_target_array(baseline_pred_scaled, scaler, target_col_index=0)
baseline_flat = baseline_unscaled.reshape(-1)
baseline_mae = mean_absolute_error(y_test_flat, baseline_flat)
print(f"\nBaseline persistence MAE: {baseline_mae:.6f}")

# --- 10. TRỰC QUAN HÓA KẾT QUẢ ---
print("Đang vẽ biểu đồ kết quả...")
# 10a. Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Loss Huấn luyện (Train)')
plt.plot(history.history['val_loss'], label='Loss Kiểm định (Validation)')
plt.title(f'Biểu đồ Loss qua các Epochs (Model: {MODEL_NAME})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(plot_loss_filename)
plt.show()

# 10b. So sánh dự báo (vẽ n_points_to_plot điểm đầu)
n_points_to_plot = 200
plt.figure(figsize=(8, 4))
plt.plot(y_test_unscaled.reshape(-1)[:n_points_to_plot], color='blue', linewidth=2, label='Mực nước Thực tế')
plt.plot(y_pred_for_eval.reshape(-1)[:n_points_to_plot], color='red', linestyle='--', linewidth=2, label=f'Dự báo ({MODEL_NAME})')
plt.title(f'So sánh dự báo Multi-Step ({N_OUT} giờ) trên {n_points_to_plot} điểm đầu của tập Test')
plt.xlabel('Bước thời gian (Giờ)')
plt.ylabel('Mực nước hồ (m)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_pred_filename, dpi=150, bbox_inches='tight')
plt.show()

# 10c. Sai số theo bước
plt.figure(figsize=(14, 7))
steps = range(1, N_OUT + 1)
plt.plot(steps, mae_per_step, label='MAE theo từng bước', marker='o', linestyle='-')
plt.plot(steps, rmse_per_step, label='RMSE theo từng bước', marker='x', linestyle='--')
plt.title(f'Sai số MAE/RMSE theo từng bước dự báo (Model: {MODEL_NAME})')
plt.xlabel('Bước dự báo (Giờ t+...)')
plt.ylabel('Sai số (mét)')
plt.xticks(steps, [f't+{s}' for s in steps], rotation=45, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_error_step_filename)
plt.show()

full_scaled = scaler.transform(data)
X_latest = full_scaled[-N_IN:, :].reshape(1, N_IN, NUM_FEATURES)
y_base_latest = model.predict(X_latest)
err_latest_in = np.concatenate([y_base_latest, np.repeat(step_idx, y_base_latest.shape[0], axis=0)], axis=2)
y_err_latest = error_model.predict(err_latest_in)
y_pred_scaled_latest = y_base_latest + y_err_latest
latest_baseline_scaled = np.repeat(X_latest[:, -1, 0].reshape(-1, 1, 1), N_OUT, axis=1)
y_pred_scaled_latest_abs = y_pred_scaled_latest + latest_baseline_scaled
y_pred_unscaled_latest = unscale_target_array(y_pred_scaled_latest_abs, scaler, target_col_index=0)
y_pred_unscaled_cal_latest = a * y_pred_unscaled_latest + b
y_final_latest = np.clip(y_pred_unscaled_cal_latest, target_min, target_max)
last_time = df['thoi_gian'].iloc[-1]
next_times = last_time + pd.to_timedelta(np.arange(1, N_OUT + 1), unit='H')
df_forecast = pd.DataFrame({'thoi_gian': next_times, 'du_bao_muc_nuoc_m': y_final_latest.reshape(-1)})
forecast_csv = f'du_bao_tiep_theo_{MODEL_NAME}.csv'
df_forecast.to_csv(forecast_csv, index=False)
plt.figure(figsize=(12, 5))
hist_times = df['thoi_gian'].iloc[-N_IN:]
hist_vals = df[TARGET_FEATURE].values[-N_IN:]
plt.plot(hist_times, hist_vals, color='blue', linewidth=2, label='Thực tế gần nhất')
plt.plot(next_times, y_final_latest.reshape(-1), color='red', linestyle='--', linewidth=2, label='Dự báo tiếp theo')
plt.title(f'Dự báo {N_OUT} bước tiếp theo từ cửa sổ gần nhất (Model: {MODEL_NAME})')
plt.xlabel('Thời gian')
plt.ylabel('Mực nước hồ (m)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
next_forecast_plot = f'bieu_do_du_bao_tiep_theo_{MODEL_NAME}.png'
plt.tight_layout()
plt.savefig(next_forecast_plot, dpi=150, bbox_inches='tight')
plt.show()
print(f"Đã lưu dự báo tiếp theo vào: {forecast_csv}")
print(f"Đã lưu biểu đồ dự báo tiếp theo vào: {next_forecast_plot}")

print(f"--- HOÀN THÀNH (Model: {MODEL_NAME}) ---")
