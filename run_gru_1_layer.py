import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# --- 1. CẤU HÌNH CÁC SIÊU THAM SỐ ---
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
N_IN = 72
N_OUT = 24  
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
UNITS = 128
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
MODEL_NAME = 'gru_1_layer'

# --- 2. TẢI VÀ XỬ LÝ DỮ LIỆU ---
print(f"--- BẮT ĐẦU CHẠY MÔ HÌNH: {MODEL_NAME} ---")
if not os.path.exists(DATA_FILE):
    print(f"LỖI: Không tìm thấy file {DATA_FILE}.")
    exit()

df = pd.read_csv(DATA_FILE, parse_dates=['thoi_gian'], dayfirst=True, na_values='--')
df = df.sort_values(by='thoi_gian')
df[SELECTED_FEATURES] = df[SELECTED_FEATURES].ffill()
data = df[SELECTED_FEATURES].values.astype(float)

# --- 3. PHÂN CHIA & CHUẨN HÓA ---
n = len(data)
n_test = int(n * TEST_SIZE)
n_val = int(n * VALIDATION_SIZE)
n_train = n - n_test - n_val

train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

print(f"Tổng mẫu: {n}, Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# --- 4. HÀM TẠO CỬA SỔ TRƯỢT ---
def create_sequences(data, n_in, n_out):
    X, y = [], []
    for i in range(len(data) - n_in - n_out + 1):
        X.append(data[i:(i + n_in), :])
        y.append(data[(i + n_in):(i + n_in + n_out), 0])
    return np.array(X), np.array(y)

print("Đang tạo cửa sổ trượt...")
X_train, y_train = create_sequences(train_scaled, N_IN, N_OUT)
X_val, y_val = create_sequences(val_scaled, N_IN, N_OUT)
X_test, y_test = create_sequences(test_scaled, N_IN, N_OUT)
y_train = y_train.reshape(-1, N_OUT, 1)
y_val = y_val.reshape(-1, N_OUT, 1)
y_test = y_test.reshape(-1, N_OUT, 1)
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

# --- 5. XÂY DỰNG MÔ HÌNH ---
print(f"--- Đang xây dựng mô hình: {MODEL_NAME} ---")
model = Sequential([
    Input(shape=(N_IN, NUM_FEATURES)),
    GRU(UNITS, return_sequences=False),  # dùng activation mặc định (tanh)
    Dropout(0.2),
    Dense(N_OUT, activation='sigmoid'),
    Reshape((N_OUT, 1))
])
model.compile(optimizer=Adam(learning_rate=5e-4, clipnorm=1.0), loss=tf.keras.losses.Huber(), metrics=['mae'])
model.summary()

# Xuất sơ đồ kiến trúc mô hình
model_diagram_filename = f'so_do_kien_truc_{MODEL_NAME}.png'
try:
    plot_model(model, to_file=model_diagram_filename, 
               show_shapes=True, 
               show_layer_names=True,
               rankdir='TB',  # Top to Bottom
               expand_nested=True,
               dpi=150)
    print(f"✅ Đã xuất sơ đồ kiến trúc mô hình vào '{model_diagram_filename}'")
except Exception as e:
    print(f"⚠️ Không thể xuất sơ đồ kiến trúc: {e}")
    print("   (Cần cài đặt graphviz và pydot: pip install graphviz pydot)")

# --- 6. HUẤN LUYỆN ---
model_filename = f'model_{MODEL_NAME}.h5'
plot_loss_filename = f'bieu_do_loss_{MODEL_NAME}.png'
plot_pred_filename = f'ket_qua_du_bao_{MODEL_NAME}.png'
plot_error_step_filename = f'sai_so_tung_buoc_{MODEL_NAME}.png'

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

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
print(f"✅ Đã lưu mô hình vào file '{model_filename}'")

# --- 7. ĐÁNH GIÁ ---
print(f"Đang đánh giá mô hình {MODEL_NAME} trên tập Test...")
y_pred_scaled = model.predict(X_test)
y_pred_scaled = np.clip(y_pred_scaled, 0.0, 1.0)

# ✅ ĐẢO CHUẨN HÓA CHÍNH XÁC
y_pred_unscaled = y_pred_scaled * scaler.data_range_[0] + scaler.data_min_[0]
y_test_unscaled = y_test * scaler.data_range_[0] + scaler.data_min_[0]

# --- Tổng quan ---
y_test_flat = y_test_unscaled.reshape(-1)
y_pred_flat = y_pred_unscaled.reshape(-1)
mae = mean_absolute_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
r2 = r2_score(y_test_flat, y_pred_flat)

print("\n--- KẾT QUẢ TỔNG QUAN ---")
print(f"MAE : {mae:.4f} (m)")
print(f"RMSE: {rmse:.4f} (m)")
print(f"R²  : {r2:.4f}")

# --- Sai số theo từng bước ---
print("\n--- SAI SỐ THEO TỪNG BƯỚC ---")
mae_per_step = np.mean(np.abs(y_test_unscaled - y_pred_unscaled), axis=0).flatten()
rmse_per_step = np.sqrt(np.mean(np.square(y_test_unscaled - y_pred_unscaled), axis=0)).flatten()
for i, (mae_val, rmse_val) in enumerate(zip(mae_per_step, rmse_per_step)):
    print(f"  t+{i+1}: MAE={mae_val:.4f} | RMSE={rmse_val:.4f}")

# --- Baseline (Persistence) ---
last_vals = X_test[:, -1, 0]
baseline_pred = np.repeat(last_vals.reshape(-1, 1), N_OUT, axis=1)
baseline_pred_unscaled = baseline_pred * scaler.data_range_[0] + scaler.data_min_[0]

baseline_mae = mean_absolute_error(y_test_unscaled.reshape(-1), baseline_pred_unscaled.reshape(-1))
baseline_rmse = np.sqrt(mean_squared_error(y_test_unscaled.reshape(-1), baseline_pred_unscaled.reshape(-1)))
print("\n--- BASELINE PERSISTENCE ---")
print(f"Baseline MAE : {baseline_mae:.4f} (m)")
print(f"Baseline RMSE: {baseline_rmse:.4f} (m)")

# --- 8. TRỰC QUAN HÓA ---
print("\n--- ĐANG VẼ CÁC BIỂU ĐỒ ---")
# a. Biểu đồ loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.title(f'Biểu đồ Loss qua các Epochs (Model: {MODEL_NAME})', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Huber Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_loss_filename, dpi=150, bbox_inches='tight')
print(f"✅ Đã lưu biểu đồ loss: {plot_loss_filename}")
plt.show()

# b. So sánh dự báo vs thực tế
n_points_to_plot = 200
plt.figure(figsize=(15, 7))
plt.plot(y_test_unscaled.reshape(-1)[:n_points_to_plot], 
         label='Mực nước Thực tế', color='blue', linewidth=2, alpha=0.7)
plt.plot(y_pred_unscaled.reshape(-1)[:n_points_to_plot], 
         label=f'Dự báo ({MODEL_NAME})', color='red', linestyle='--', linewidth=2)
plt.title(f'So sánh dự báo Multi-Step ({N_OUT} giờ) trên {n_points_to_plot} điểm đầu của tập Test',
         fontsize=14, fontweight='bold')
plt.xlabel('Bước thời gian (Giờ)', fontsize=12)
plt.ylabel('Mực nước hồ (m)', fontsize=12)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_pred_filename, dpi=150, bbox_inches='tight')
print(f"✅ Đã lưu biểu đồ dự báo: {plot_pred_filename}")
plt.show()

# c. Sai số theo từng bước
plt.figure(figsize=(14, 7))
steps = np.arange(1, N_OUT + 1)
plt.plot(steps, mae_per_step, label='MAE theo từng bước', marker='o', 
         linestyle='-', linewidth=2, markersize=6)
plt.plot(steps, rmse_per_step, label='RMSE theo từng bước', marker='x', 
         linestyle='--', linewidth=2, markersize=6)
plt.title(f'Sai số MAE/RMSE theo từng bước dự báo (Model: {MODEL_NAME})',
         fontsize=14, fontweight='bold')
plt.xlabel('Bước dự báo (Giờ t+...)', fontsize=12)
plt.ylabel('Sai số (mét)', fontsize=12)
plt.xticks(steps[::2], [f't+{s}' for s in steps[::2]], rotation=45, ha='right')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_error_step_filename, dpi=150, bbox_inches='tight')
print(f"✅ Đã lưu biểu đồ sai số theo bước: {plot_error_step_filename}")
plt.show()

# d. Biểu đồ phân tán (Scatter plot) - Thực tế vs Dự báo
scatter_filename = f'phan_tan_thuc_te_vs_du_bao_{MODEL_NAME}.png'
plt.figure(figsize=(10, 10))
plt.scatter(y_test_flat, y_pred_flat, alpha=0.5, s=10)
plt.plot([y_test_flat.min(), y_test_flat.max()], 
         [y_test_flat.min(), y_test_flat.max()], 
         'r--', linewidth=2, label='Đường lý tưởng')
plt.title(f'Biểu đồ phân tán: Giá trị Thực tế vs Dự báo (Model: {MODEL_NAME})',
         fontsize=14, fontweight='bold')
plt.xlabel('Mực nước Thực tế (m)', fontsize=12)
plt.ylabel('Mực nước Dự báo (m)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig(scatter_filename, dpi=150, bbox_inches='tight')
print(f"✅ Đã lưu biểu đồ phân tán: {scatter_filename}")
plt.show()

# e. Biểu đồ phân phối sai số
error_dist_filename = f'phan_phoi_sai_so_{MODEL_NAME}.png'
errors = y_test_flat - y_pred_flat
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Sai số = 0')
plt.title('Histogram phân phối sai số', fontsize=12, fontweight='bold')
plt.xlabel('Sai số (m)', fontsize=11)
plt.ylabel('Tần suất', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(errors, vert=True)
plt.title('Boxplot phân phối sai số', fontsize=12, fontweight='bold')
plt.ylabel('Sai số (m)', fontsize=11)
plt.grid(True, alpha=0.3)

plt.suptitle(f'Phân phối sai số dự báo (Model: {MODEL_NAME})', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(error_dist_filename, dpi=150, bbox_inches='tight')
print(f"✅ Đã lưu biểu đồ phân phối sai số: {error_dist_filename}")
plt.show()

print("\n" + "="*60)
print(f"✅ HOÀN THÀNH (Model: {MODEL_NAME})")
print("="*60)
print("\n📊 CÁC FILE ĐÃ XUẤT:")
print(f"  1. Mô hình: {model_filename}")
print(f"  2. Sơ đồ kiến trúc: {model_diagram_filename}")
print(f"  3. Biểu đồ loss: {plot_loss_filename}")
print(f"  4. Biểu đồ dự báo: {plot_pred_filename}")
print(f"  5. Biểu đồ sai số theo bước: {plot_error_step_filename}")
print(f"  6. Biểu đồ phân tán: {scatter_filename}")
print(f"  7. Biểu đồ phân phối sai số: {error_dist_filename}")
print("="*60)
