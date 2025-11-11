import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. CẤU HÌNH ---
DATA_FILE = 'HoaBinh.csv'
COLUMNS_TO_PLOT = [
    'Mực nước hồ (m)',
    'Lưu lượng đến hồ (m³/s)',
    'Tổng lưu lượng xả (m³/s)[Thực tế]'
]
OUTPUT_IMAGE_FILE = 'Hinh_3_1_truc_quan_hoa.png'

# --- 2. KIỂM TRA FILE ---
if not os.path.exists(DATA_FILE):
    print(f"LỖI: Không tìm thấy file {DATA_FILE}. Vui lòng kiểm tra lại.")
    exit()

# --- 3. TẢI VÀ XỬ LÝ DỮ LIỆU ---
print("Đang tải dữ liệu...")
df = pd.read_csv(DATA_FILE, parse_dates=['thoi_gian'], dayfirst=True, na_values='--')

# Sắp xếp theo thời gian và đặt làm chỉ mục (index) để vẽ biểu đồ
df = df.sort_values(by='thoi_gian')
df = df.set_index('thoi_gian')

# Chọn các cột để vẽ và xử lý (ffill)
data_to_plot = df[COLUMNS_TO_PLOT]
data_to_plot = data_to_plot.ffill()

# --- 4. VẼ BIỂU ĐỒ ---
print("Đang vẽ biểu đồ...")
# Sử dụng subplots=True vì 3 cột có thang đo (scale) rất khác nhau
# (Mực nước ~100, trong khi Lưu lượng ~1000s)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10), sharex=True)

# Đặt tiêu đề chung cho cả 3 biểu đồ
fig.suptitle('Trực quan hóa dữ liệu vận hành Hồ Hòa Bình', fontsize=16)

# Vẽ từng biểu đồ con
data_to_plot[COLUMNS_TO_PLOT[0]].plot(ax=axes[0], title=COLUMNS_TO_PLOT[0], color='blue')
axes[0].set_ylabel('Mét (m)')

data_to_plot[COLUMNS_TO_PLOT[1]].plot(ax=axes[1], title=COLUMNS_TO_PLOT[1], color='green')
axes[1].set_ylabel('m³/s')

data_to_plot[COLUMNS_TO_PLOT[2]].plot(ax=axes[2], title=COLUMNS_TO_PLOT[2], color='red')
axes[2].set_ylabel('m³/s')

# Đặt nhãn trục X cho biểu đồ cuối cùng
axes[2].set_xlabel('Thời gian (Năm-Tháng)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Điều chỉnh tiêu đề chung

# Lưu file ảnh
plt.savefig(OUTPUT_IMAGE_FILE)
print(f"Đã lưu biểu đồ vào file: {OUTPUT_IMAGE_FILE}")

# Hiển thị biểu đồ
plt.show()

print("--- HOÀN THÀNH ---")