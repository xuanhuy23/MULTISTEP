import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import joblib 

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
MODEL_NAME = 'gru_2_layer_hybrid' # Tên mô hình lai

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
joblib.dump(scaler, f'scaler_{MODEL_NAME}.pkl') 
print(f"Đã lưu scaler vào file 'scaler_{MODEL_NAME}.pkl'")

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

# --- 5. HÀM XÂY DỰNG MÔ HÌNH GRU 2 LỚP (TÁI SỬ DỤNG) ---
def build_gru_2_layer_model(units, n_in, n_out, num_features):
    """
    Hàm này xây dựng kiến trúc GRU 2 Lớp (Stacked GRU)
    Chúng ta sẽ gọi hàm này 2 LẦN:
    1. Cho model_main (dự báo mực nước)
    2. Cho model_residual (dự báo sai số)
    """
    model = Sequential()
    model.add(Input(shape=(n_in, num_features)))
    # Lớp GRU thứ nhất
    model.add(GRU(units, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    # Lớp GRU thứ hai
    model.add(GRU(units, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    # Lớp Dense (Decoder)
    model.add(Dense(n_out, activation='linear'))
    model.add(Reshape((n_out, 1)))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- 6. GIAI ĐOẠN 1: HUẤN LUYỆN MÔ HÌNH CHÍNH (model_main) ---
print("\n--- BẮT ĐẦU GIAI ĐOẠN 1: HUẤN LUYỆN MÔ HÌNH CHÍNH ---")
model_main = build_gru_2_layer_model(UNITS, N_IN, N_OUT, NUM_FEATURES)
model_main.summary()

model_main_filename = 'model_gru_main.h5'
plot_loss_main_filename = 'bieu_do_loss_main.png'
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history_main = model_main.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)
model_main.save(model_main_filename)
print(f"Đã lưu mô hình Giai đoạn 1 vào file '{model_main_filename}'")

# --- 7. TẠO DỮ LIỆU MỚI (SAI SỐ) CHO GIAI ĐOẠN 2 ---
print("\n--- CHUẨN BỊ GIAI ĐOẠN 2: TÍNH TOÁN SAI SỐ (LOSS 1) ---")

# 1. Tính toán sai số trên tập Train (để tạo y_train mới)
y_pred_train_scaled = model_main.predict(X_train)
y_residuals_train_scaled = y_train - y_pred_train_scaled # y_thực_tế - y_dự_báo
print(f"Shape của dữ liệu sai số (Train): {y_residuals_train_scaled.shape}")

# 2. Tính toán sai số trên tập Validation (để tạo y_val mới)
y_pred_val_scaled = model_main.predict(X_val)
y_residuals_val_scaled = y_val - y_pred_val_scaled
print(f"Shape của dữ liệu sai số (Validation): {y_residuals_val_scaled.shape}")

# Lưu ý: X_train và X_val vẫn được giữ nguyên làm đầu vào

# --- 8. GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH HIỆU CHỈNH SAI SỐ (model_residual) ---
print("\n--- BẮT ĐẦU GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH HIỆU CHỈNH SAI SỐ ---")
# Chúng ta dùng chung kiến trúc, nhưng đây là một mô hình hoàn toàn MỚI
model_residual = build_gru_2_layer_model(UNITS, N_IN, N_OUT, NUM_FEATURES)
model_residual.summary()

model_residual_filename = 'model_gru_residual.h5'
plot_loss_residual_filename = 'bieu_do_loss_residual.png'

# Huấn luyện model_residual để HỌC CÁCH DỰ BÁO SAI SỐ
history_residual = model_residual.fit(
    X_train, y_residuals_train_scaled,  # X là X_train, Y là SAI SỐ (Loss 1)
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_residuals_val_scaled), # Val cũng tương tự
    callbacks=[early_stopping],
    verbose=1
)
model_residual.save(model_residual_filename)
print(f"Đã lưu mô hình Giai đoạn 2 vào file '{model_residual_filename}'")


# --- 9. ĐÁNH GIÁ MÔ HÌNH LAI (HYBRID) TRÊN TẬP TEST ---
print(f"\n--- ĐÁNH GIÁ MÔ HÌNH LAI ({MODEL_NAME}) TRÊN TẬP TEST ---")

# 1. Lấy dự báo từ mô hình 1 (Dự báo chính)
y_pred_main_scaled = model_main.predict(X_test)
# 2. Lấy dự báo từ mô hình 2 (Dự báo sai số)
y_pred_residual_scaled = model_residual.predict(X_test)

# 3. KẾT QUẢ CUỐI CÙNG = Dự báo chính + Dự báo sai số
y_pred_final_scaled = y_pred_main_scaled + y_pred_residual_scaled

# 4. Đảo ngược chuẩn hóa (Inverse Scaling)
target_scaler_min = scaler.min_[0]
target_scaler_scale = scaler.scale_[0] 
# Đảo ngược cho dự báo cuối cùng
y_pred_unscaled = (y_pred_final_scaled * target_scaler_scale) + target_scaler_min
# Đảo ngược cho giá trị thực tế (để so sánh)
y_test_unscaled = (y_test * target_scaler_scale) + target_scaler_min

# --- 9a. Đánh giá TỔNG QUAN ---
y_test_flat = y_test_unscaled.reshape(-1)
y_pred_flat = y_pred_unscaled.reshape(-1)
mae = mean_absolute_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
r2 = r2_score(y_test_flat, y_pred_flat)

print("\n--- KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN (MÔ HÌNH LAI) ---")
print(f"Mô hình: {MODEL_NAME}")
print(f"Chỉ số MAE (Sai số tuyệt đối trung bình): {mae:.4f} (mét)")
print(f"Chỉ số RMSE (Căn bậc hai sai số TB): {rmse:.4f} (mét)")
print(f"Chỉ số R² (Hệ số xác định): {r2:.4f}")

# --- 9b. Đánh giá SAI SỐ THEO TỪNG BƯỚC DỰ BÁO ---
print("\n--- KẾT QUẢ SAI SỐ THEO TỪNG BƯỚC DỰ BÁO (MÔ HÌNH LAI) ---")
mae_per_step = np.mean(np.abs(y_test_unscaled - y_pred_unscaled), axis=0).flatten()
rmse_per_step = np.sqrt(np.mean(np.square(y_test_unscaled - y_pred_unscaled), axis=0)).flatten()
for i, (mae_val, rmse_val) in enumerate(zip(mae_per_step, rmse_per_step)):
    print(f"  Giờ t+{i+1}: MAE={mae_val:.4f} (mét) | RMSE={rmse_val:.4f} (mét)")

# --- 10. TRỰC QUAN HÓA KẾT QUẢ ---
print("Đang vẽ biểu đồ kết quả...")

# 10a. Trực quan hóa loss MÔ HÌNH CHÍNH
plt.figure(figsize=(12, 6))
plt.plot(history_main.history['loss'], label='Loss Huấn luyện (Train)')
plt.plot(history_main.history['val_loss'], label='Loss Kiểm định (Validation)')
plt.title(f'Biểu đồ Loss (Giai đoạn 1: Mô hình chính)')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(plot_loss_main_filename)
plt.show()

# 10b. Trực quan hóa loss MÔ HÌNH SAI SỐ
plt.figure(figsize=(12, 6))
plt.plot(history_residual.history['loss'], label='Loss Huấn luyện (Train - Sai số)')
plt.plot(history_residual.history['val_loss'], label='Loss Kiểm định (Validation - Sai số)')
plt.title(f'Biểu đồ Loss (Giai đoạn 2: Mô hình hiệu chỉnh sai số)')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(plot_loss_residual_filename)
plt.show()

# 10c. Trực quan hóa KẾT QUẢ CUỐI CÙNG (LAI)
plot_pred_filename = f'ket_qua_du_bao_{MODEL_NAME}.png'
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

# 10d. Trực quan hóa SAI SỐ CUỐI CÙNG (LAI)
plot_error_step_filename = f'sai_so_tung_buoc_{MODEL_NAME}.png'
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