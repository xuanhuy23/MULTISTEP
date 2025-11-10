import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Reshape, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# --- 1. CẤU HÌNH CÁC SIÊU THAM SỐ (Hyperparameters) ---
N_IN = 72
N_OUT = 24
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
UNITS = 128 # Số units trong MỖI lớp GRU
EPOCHS = 100
BATCH_SIZE = 64
DATA_FILE = 'HoaBinh.csv'
SELECTED_FEATURES = [
    'Mực nước hồ (m)',
    'Lưu lượng đến hồ (m³/s)',
    'Tổng lưu lượng xả (m³/s)[Thực tế]'
]
TARGET_FEATURE = 'Mực nước hồ (m)'
NUM_FEATURES = len(SELECTED_FEATURES)

# Định nghĩa tên mô hình cho file này
MODEL_NAME = 'gru_2_layer'

# --- 2. TẢI VÀ XỬ LÝ DỮ LIỆU ---
print(f"--- BẮT ĐẦU CHẠY MÔ HÌNH: {MODEL_NAME} ---")
print("Đang tải dữ liệu...")
if not os.path.exists(DATA_FILE):
    print(f"LỖI: Không tìm thấy file {DATA_FILE}. Vui lòng kiểm tra lại.")
    exit()
    
df = pd.read_csv(DATA_FILE, parse_dates=['thoi_gian'], dayfirst=True, na_values='--')
df = df.sort_values(by='thoi_gian')
df[SELECTED_FEATURES] = df[SELECTED_FEATURES].ffill()
data = df[SELECTED_FEATURES].values.astype(float)

# --- 3. PHÂN CHIA VÀ CHUẨN HÓA DỮ LIỆU ---
n = len(data)
n_test = int(n * TEST_SIZE)
n_val = int(n * VALIDATION_SIZE)
n_train = n - n_test - n_val
train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]
print(f"Tổng số mẫu: {n}, Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# --- 4. HÀM TẠO CỬA SỔ TRƯỢT (SLIDING WINDOW) ---
def create_sequences(data, n_in, n_out):
    X, y = [], []
    for i in range(len(data) - n_in - n_out + 1):
        X.append(data[i:(i + n_in), :])
        y.append(data[(i + n_in):(i + n_in + n_out), 0])
    return np.array(X), np.array(y)

print("Đang tạo mẫu cửa sổ trượt (windowing)...")
X_train, y_train = create_sequences(train_scaled, N_IN, N_OUT)
X_val, y_val = create_sequences(val_scaled, N_IN, N_OUT)
X_test, y_test = create_sequences(test_scaled, N_IN, N_OUT)
y_train = y_train.reshape(-1, N_OUT, 1)
y_val = y_val.reshape(-1, N_OUT, 1)
y_test = y_test.reshape(-1, N_OUT, 1)
print(f"Shape của X_train: {X_train.shape}, Shape của y_train: {y_train.shape}")

# --- 5. XÂY DỰNG MÔ HÌNH GRU 2 LỚP ---
print(f"--- Đang xây dựng mô hình: {MODEL_NAME} ---")
model = Sequential()
model.add(Input(shape=(N_IN, NUM_FEATURES)))
# 4. Mô hình GRU 2 lớp (Stacked GRU)
# Lớp GRU thứ nhất: Phải trả về chuỗi (return_sequences=True)
model.add(GRU(UNITS, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
# Lớp GRU thứ hai: Chỉ trả về trạng thái cuối (return_sequences=False)
model.add(GRU(UNITS, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
# Lớp Dense (Decoder)
model.add(Dense(N_OUT, activation='linear'))
model.add(Reshape((N_OUT, 1)))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 6. HUẤN LUYỆN MÔ HÌNH ---
model_filename = f'model_{MODEL_NAME}.h5'
plot_loss_filename = f'bieu_do_loss_{MODEL_NAME}.png'
plot_pred_filename = f'ket_qua_du_bao_{MODEL_NAME}.png'
plot_error_step_filename = f'sai_so_tung_buoc_{MODEL_NAME}.png'

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print(f"Đang huấn luyện mô hình {MODEL_NAME}...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)
model.save(model_filename)
print(f"Đã lưu mô hình vào file '{model_filename}'")

# --- 7. ĐÁNH GIÁ MÔ HÌNH ---
print(f"Đang đánh giá mô hình {MODEL_NAME} trên tập Test...")
y_pred_scaled = model.predict(X_test)
target_scaler_min = scaler.min_[0]
target_scaler_scale = scaler.scale_[0] 
y_pred_unscaled = (y_pred_scaled * target_scaler_scale) + target_scaler_min
y_test_unscaled = (y_test * target_scaler_scale) + target_scaler_min

# --- 7a. Đánh giá TỔNG QUAN ---
y_test_flat = y_test_unscaled.reshape(-1)
y_pred_flat = y_pred_unscaled.reshape(-1)
mae = mean_absolute_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
r2 = r2_score(y_test_flat, y_pred_flat)

print("\n--- KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN ---")
print(f"Mô hình: {MODEL_NAME}")
print(f"Chỉ số MAE (Sai số tuyệt đối trung bình): {mae:.4f} (mét)")
print(f"Chỉ số RMSE (Căn bậc hai sai số TB): {rmse:.4f} (mét)")
print(f"Chỉ số R² (Hệ số xác định): {r2:.4f}")

# --- 7b. Đánh giá SAI SỐ THEO TỪNG BƯỚC DỰ BÁO ---
print("\n--- KẾT QUẢ SAI SỐ THEO TỪNG BƯỚC DỰ BÁO (HORIZON) ---")
mae_per_step = np.mean(np.abs(y_test_unscaled - y_pred_unscaled), axis=0).flatten()
rmse_per_step = np.sqrt(np.mean(np.square(y_test_unscaled - y_pred_unscaled), axis=0)).flatten()
for i, (mae_val, rmse_val) in enumerate(zip(mae_per_step, rmse_per_step)):
    print(f"  Giờ t+{i+1}: MAE={mae_val:.4f} (mét) | RMSE={rmse_val:.4f} (mét)")

# --- 8. TRỰC QUAN HÓA KẾT QUẢ ---
# 8a. Trực quan hóa quá trình huấn luyện
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Loss Huấn luyện (Train)')
plt.plot(history.history['val_loss'], label='Loss Kiểm định (Validation)')
plt.title(f'Biểu đồ Loss qua các Epochs (Model: {MODEL_NAME})')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(plot_loss_filename)
plt.show()

# 8b. Trực quan hóa kết quả dự báo
n_points_to_plot = 200
plt.figure(figsize=(15, 7))
plt.plot(y_test_unscaled.reshape(-1)[:n_points_to_plot], label='Mực nước Thực tế', color='blue', marker='.')
plt.plot(y_pred_unscaled.reshape(-1)[:n_points_to_plot], label=f'Dự báo ({MODEL_NAME})', color='red', linestyle='--')
plt.title(f'So sánh dự báo Multi-Step ({N_OUT} giờ) trên {n_points_to_plot} điểm đầu của tập Test')
plt.xlabel('Bước thời gian (Giờ)')
plt.ylabel('Mực nước hồ (m)')
plt.legend()
plt.grid(True)
plt.savefig(plot_pred_filename)
plt.show()

# 8c. Trực quan hóa SAI SỐ THEO TỪNG BƯỚC DỰ BÁO
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

print(f"--- HOÀN THÀNH (Model: {MODEL_NAME}) ---")