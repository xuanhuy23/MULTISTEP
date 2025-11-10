import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH CÁC SIÊU THAM SỐ (Hyperparameters) ---
# (Bạn có thể tinh chỉnh các giá trị này theo Chương 3)

# Số bước thời gian quá khứ dùng để dự báo (ví dụ: 72 giờ = 3 ngày)
N_IN = 72
# Số bước thời gian tương lai cần dự báo (ví dụ: 24 giờ = 1 ngày)
N_OUT = 24
# Tỷ lệ dữ liệu cho tập Test (ví dụ: 15%)
TEST_SIZE = 0.15
# Tỷ lệ dữ liệu cho tập Validation (ví dụ: 15%)
VALIDATION_SIZE = 0.15
# Số units trong lớp GRU (số tế bào bộ nhớ)
GRU_UNITS = 128
# Các tham số huấn luyện
EPOCHS = 100
BATCH_SIZE = 64
# File dữ liệu
DATA_FILE = 'HoaBinh.csv'
# Các cột đặc trưng (features) và mục tiêu (target)
# CỘT ĐẦU TIÊN PHẢI LÀ MỤC TIÊU CẦN DỰ BÁO (Mực nước hồ)
SELECTED_FEATURES = [
    'Mực nước hồ (m)',
    'Lưu lượng đến hồ (m³/s)',
    'Tổng lưu lượng xả (m³/s)[Thực tế]'
]
TARGET_FEATURE = 'Mực nước hồ (m)'
NUM_FEATURES = len(SELECTED_FEATURES)

# --- 2. TẢI VÀ XỬ LÝ DỮ LIỆU ---

print("Đang tải dữ liệu...")
# Đọc dữ liệu, chỉ định cột 'thoi_gian' là datetime
df = pd.read_csv(DATA_FILE, parse_dates=['thoi_gian'], dayfirst=True, na_values='--')

# Sắp xếp theo thời gian (quan trọng)
df = df.sort_values(by='thoi_gian')

# Xử lý giá trị thiếu (nếu có) - ở đây dùng 'ffill' (điền giá trị trước đó)
df[SELECTED_FEATURES] = df[SELECTED_FEATURES].ffill()

# Chọn các cột đã định nghĩa
data = df[SELECTED_FEATURES].values.astype(float)

# --- 3. PHÂN CHIA VÀ CHUẨN HÓA DỮ LIỆU ---

# Tính toán kích thước các tập
n = len(data)
n_test = int(n * TEST_SIZE)
n_val = int(n * VALIDATION_SIZE)
n_train = n - n_test - n_val

# Phân chia theo thứ tự thời gian (không xáo trộn)
train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

print(f"Tổng số mẫu: {n}")
print(f"Tập Huấn luyện (Train): {len(train_data)} mẫu")
print(f"Tập Kiểm định (Validation): {len(val_data)} mẫu")
print(f"Tập Thử nghiệm (Test): {len(test_data)} mẫu")

# Chuẩn hóa (Scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
# CHỈ FIT trên tập Train
scaler = scaler.fit(train_data)

# TRANSFORM cho cả 3 tập
train_scaled = scaler.transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# --- 4. HÀM TẠO CỬA SỔ TRƯỢT (SLIDING WINDOW) ---

def create_sequences(data, n_in, n_out):
    """
    Hàm này chuyển đổi chuỗi thời gian thành các mẫu (X, Y)
    cho bài toán dự báo multi-step (Seq2Seq).
    """
    X, y = [], []
    for i in range(len(data) - n_in - n_out + 1):
        # Đầu vào: (i) đến (i + n_in)
        # Bao gồm TẤT CẢ các đặc trưng
        X.append(data[i:(i + n_in), :])
        
        # Đầu ra: (i + n_in) đến (i + n_in + n_out)
        # CHỈ LẤY cột mục tiêu (cột 0, 'Mực nước hồ (m)')
        y.append(data[(i + n_in):(i + n_in + n_out), 0])
        
    return np.array(X), np.array(y)

print("Đang tạo mẫu cửa sổ trượt (windowing)...")
X_train, y_train = create_sequences(train_scaled, N_IN, N_OUT)
X_val, y_val = create_sequences(val_scaled, N_IN, N_OUT)
X_test, y_test = create_sequences(test_scaled, N_IN, N_OUT)

# Reshape y để phù hợp với đầu ra của mô hình (thêm 1 chiều)
# y_train, y_val, y_test sẽ có shape (samples, N_OUT, 1)
y_train = y_train.reshape(-1, N_OUT, 1)
y_val = y_val.reshape(-1, N_OUT, 1)
y_test = y_test.reshape(-1, N_OUT, 1)

print(f"Shape của X_train: {X_train.shape}")
print(f"Shape của y_train: {y_train.shape}")
print(f"Shape của X_test: {X_test.shape}")
print(f"Shape của y_test: {y_test.shape}")

# --- 5. XÂY DỰNG MÔ HÌNH GRU MULTI-OUTPUT ---

print("Đang xây dựng mô hình GRU...")
model = Sequential()

# Lớp Input
model.add(Input(shape=(N_IN, NUM_FEATURES)))

# Lớp GRU (Encoder)
# return_sequences=False vì chỉ lấy trạng thái ẩn cuối cùng
model.add(GRU(GRU_UNITS, return_sequences=False, activation='relu'))
model.add(Dropout(0.2)) # Chống quá khớp (overfitting)

# Lớp Dense (Decoder)
# units = N_OUT để dự báo N_OUT bước đồng thời
model.add(Dense(N_OUT, activation='linear')) # 'linear' cho bài toán hồi quy

# Reshape đầu ra về (N_OUT, 1) để khớp với y_test
model.add(Reshape((N_OUT, 1)))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='mean_squared_error') # Dùng MSE làm hàm mất mát
model.summary()

# --- 6. HUẤN LUYỆN MÔ HÌNH ---

# Sử dụng EarlyStopping để dừng khi val_loss không cải thiện
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Đang huấn luyện mô hình...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Lưu mô hình đã huấn luyện (quan trọng cho luận văn)
model.save('gru_model_hoso_hoabinh.h5')
print("Đã lưu mô hình vào file 'gru_model_hoso_hoabinh.h5'")

# --- 7. ĐÁNH GIÁ MÔ HÌNH ---

print("Đang đánh giá mô hình trên tập Test...")
# Dự báo trên tập Test
y_pred_scaled = model.predict(X_test)

# Lấy các giá trị min và scale của CHỈ CỘT MỤC TIÊU (cột 0)
# Đây là cách để đảo ngược chuẩn hóa (inverse transform) một cách chính xác
target_scaler_min = scaler.min_[0]
target_scaler_scale = scaler.scale_[0] # (max - min)

# Đảo ngược chuẩn hóa (Inverse Scaling)
# Công thức: unscaled = scaled * (max - min) + min
y_pred_unscaled = (y_pred_scaled * target_scaler_scale) + target_scaler_min
y_test_unscaled = (y_test * target_scaler_scale) + target_scaler_min

# Tính toán các chỉ số lỗi trên dữ liệu gốc (chưa chuẩn hóa)
# Cần reshape lại để tính toán (làm phẳng chuỗi 24 giờ)
y_test_flat = y_test_unscaled.reshape(-1)
y_pred_flat = y_pred_unscaled.reshape(-1)

mae = mean_absolute_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
r2 = r2_score(y_test_flat, y_pred_flat)

print("\n--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ---")
print(f"Chỉ số MAE (Sai số tuyệt đối trung bình): {mae:.4f} (mét)")
print(f"Chỉ số RMSE (Căn bậc hai sai số TB): {rmse:.4f} (mét)")
print(f"Chỉ số R² (Hệ số xác định): {r2:.4f}")

# --- 8. TRỰC QUAN HÓA KẾT QUẢ ---

print("Đang vẽ biểu đồ kết quả...")
# Trực quan hóa quá trình huấn luyện
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Loss Huấn luyện (Train)')
plt.plot(history.history['val_loss'], label='Loss Kiểm định (Validation)')
plt.title('Biểu đồ Loss qua các Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig('bieu_do_loss.png')
plt.show()

# Trực quan hóa kết quả dự báo (so sánh 200 điểm đầu tiên của tập Test)
n_points_to_plot = 200
plt.figure(figsize=(15, 7))
plt.plot(y_test_unscaled.reshape(-1)[:n_points_to_plot], label='Mực nước Thực tế', color='blue', marker='.')
plt.plot(y_pred_unscaled.reshape(-1)[:n_points_to_plot], label='Mực nước Dự báo (GRU)', color='red', linestyle='--')
plt.title(f'So sánh dự báo GRU Multi-Step (24 giờ) trên {n_points_to_plot} điểm đầu của tập Test')
plt.xlabel('Bước thời gian (Giờ)')
plt.ylabel('Mực nước hồ (m)')
plt.legend()
plt.grid(True)
plt.savefig('ket_qua_du_bao.png')
plt.show()

print("--- HOÀN THÀNH ---")