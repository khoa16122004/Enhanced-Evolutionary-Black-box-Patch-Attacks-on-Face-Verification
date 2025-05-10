import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os # Import thư viện os để quản lý đường dẫn

# --- 1. Data Preparation ---
# Tạo DataFrame từ dữ liệu trong ảnh (ĐÃ CẬP NHẬT DATA MANTIS NEW PIPELINE)
data = {
    'Model': [
        'LLaVA One vision', 'LLaVA One vision (new pipeline)', 'LLaVA One vision (direct return)',
        'LLaVA Interleave', 'LLaVA Interleave (new pipeline)', 'LLaVA Interleave (direct return)',
        'Deepseek', 'Deepseek (new pipeline)', 'Deepseek (direct return)',
        'Mantis', 'Mantis (new pipeline)', 'Mantis (direct return)',
        'ChatGPT 4o',
        'Facenet: restnet_vggface', 'Facenet: restnet_webface'
    ],
    'Label 0 (the same)': [
        0.892, 0.9523, 0.9996,
        0.9926, 0.95, 1,
        0.819, 0.832, 0.9996,
        0.9346, 0.892, 0.8977, # <<< UPDATED Mantis (new pipeline)
        'In proccess',
        0.949, 0.898
    ],
    'Label 1 (the difference)': [
        0.8726, 0.8583, 0.689,
        0.1886, 0.684, 0,
        0.2043, 0.5107, 0,
        0.127, 0.371, 0.1183, # <<< CONFIRMED Mantis (new pipeline)
        'In proccess',
        0.999, 0.9983
    ],
    'Average': [
        0.8816, 0.9053, 0.83867,
        0.5903, 0.817, 0.5,
        0.42026, 0.6713, 0.4998,
        0.5308, 0.6315, 0.508, # <<< UPDATED Mantis (new pipeline)
        'In proccess',
        0.974, 0.9481
    ],
    'Macro F1 score': [
        0.9375, 0.9492, 0.9079,
        0.6568, 0.8125, 0.5,
        0.6196, 0.663, 0.4999,
        0.5953, 0.6042, 0.5792, # <<< UPDATED Mantis (new pipeline)
        'In proccess',
        0.9863, 0.9728
    ]
}

df = pd.DataFrame(data)

# Chuyển các cột điểm số sang dạng số, bỏ qua các giá trị 'In proccess' (sẽ thành NaN)
score_cols = ['Label 0 (the same)', 'Label 1 (the difference)', 'Average', 'Macro F1 score']
for col in score_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Thêm cột Model Family và Pipeline Type để dễ nhóm và lọc
def categorize_model(model_name):
    if 'LLaVA One vision' in model_name:
        family = 'LLaVA One vision'
    elif 'LLaVA Interleave' in model_name:
        family = 'LLaVA Interleave'
    elif 'Deepseek' in model_name:
        family = 'Deepseek'
    elif 'Mantis' in model_name:
        family = 'Mantis'
    elif 'Facenet' in model_name:
        family = 'Facenet'
    elif 'ChatGPT' in model_name:
        family = 'ChatGPT'
    else:
        family = 'Other'

    if '(new pipeline)' in model_name:
        pipeline = 'New Pipeline'
    elif '(direct return)' in model_name:
        pipeline = 'Direct Return'
    elif family in ['LLaVA One vision', 'LLaVA Interleave', 'Deepseek', 'Mantis']: # Các model gốc không có hậu tố
         pipeline = 'Original'
    elif family in ['Facenet', 'ChatGPT']: # Các model khác không có hậu tố pipeline của bạn
         pipeline = 'Baseline/Other'
    else:
        pipeline = 'Unknown'

    return family, pipeline

df[['Model Family', 'Pipeline Type']] = df['Model'].apply(lambda x: pd.Series(categorize_model(x)))

# Loại bỏ các dòng có kết quả 'In proccess' cho việc vẽ biểu đồ số
df_plottable = df.dropna(subset=score_cols).copy()

# --- Custom color palette for pipeline types ---
pipeline_colors = {
    'Original': 'skyblue',
    'Direct Return': 'lightcoral',
    'New Pipeline': 'mediumseagreen', # Màu xanh nổi bật cho pipeline mới
    'Baseline/Other': 'gold' # Màu vàng cho baseline
}

# Tạo thư mục để lưu ảnh nếu chưa tồn tại
output_dir = 'benchmark_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"Biểu đồ sẽ được lưu vào thư mục: {output_dir}")


# --- 2. Biểu đồ Phân tán (Scatter Plot) cho VLM mã nguồn mở (Bias Insight) ---
# Chỉ lấy các phiên bản Original và Direct Return của LLaVA, Deepseek, Mantis
df_bias = df_plottable[
    df_plottable['Pipeline Type'].isin(['Original', 'Direct Return']) &
    df_plottable['Model Family'].isin(['LLaVA One vision', 'LLaVA Interleave', 'Deepseek', 'Mantis'])
].copy()

plt.figure(figsize=(10, 8))
scatter_bias = sns.scatterplot(data=df_bias, x='Label 0 (the same)', y='Label 1 (the difference)',
                               hue='Model Family', style='Pipeline Type', s=150)

# Thêm đường y = x
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Balanced (y=x)', alpha=0.7)

# Thêm chú thích cho từng điểm (Model Name) - Tinh chỉnh vị trí nhãn
for i, row in df_bias.iterrows():
    label = row['Model'].replace('LLaVA One vision', 'LLaVA OV').replace('LLaVA Interleave', 'LLaVA IL').replace('Deepseek', 'DS').replace('Mantis', 'MT').replace('(direct return)', '(DR)')
    x_pos, y_pos = row['Label 0 (the same)'], row['Label 1 (the difference)']

    # --- THAY ĐỔI: Logic đặt nhãn dựa trên vị trí ---
    # Đặt nhãn hơi xuống dưới và sang phải nếu L0 cao (gần mép phải)
    if x_pos > 0.9: # Ngưỡng L0 cao
         plt.text(x_pos + 0.015, y_pos - 0.01, label, fontsize=9, ha='left', va='top') # Đặt hơi xuống, căn theo đỉnh nhãn
    else: # Các điểm khác, đặt hơi lên trên và sang phải
         plt.text(x_pos + 0.015, y_pos + 0.01, label, fontsize=9, ha='left', va='bottom') # Đặt hơi lên, căn theo đáy nhãn


plt.title('Bias towards "Same" (Label 0) in Original/Direct Return Open Source VLMs')
plt.xlabel('Label 0 (the same) Score')
plt.ylabel('Label 1 (the difference) Score')
plt.xlim(0, 1.05) # Có thể nới rộng giới hạn trục nếu cần
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend(title='Model Family / Pipeline Type', loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_bias_scatterplot.png'), dpi=300, bbox_inches='tight') # Lưu ảnh
plt.show()


# --- 3. Biểu đồ Phân tán (Scatter Plot) so sánh Gốc và New Pipeline (New Pipeline Insight) ---
# Chỉ lấy các phiên bản Original và New Pipeline của LLaVA, Deepseek, Mantis
df_pipeline_effect = df_plottable[
    df_plottable['Pipeline Type'].isin(['Original', 'New Pipeline']) &
    df_plottable['Model Family'].isin(['LLaVA One vision', 'LLaVA Interleave', 'Deepseek', 'Mantis'])
].copy()

plt.figure(figsize=(10, 8))
scatter_effect_ax = sns.scatterplot(data=df_pipeline_effect, x='Label 0 (the same)', y='Label 1 (the difference)',
                                 hue='Model Family', style='Pipeline Type', s=150)

# --- Tạo color map dựa trên palette mà seaborn sử dụng cho 'Model Family' ---
model_families_effect = df_pipeline_effect['Model Family'].unique()
family_palette_effect = sns.color_palette("deep", len(model_families_effect))
color_map_effect = dict(zip(model_families_effect, family_palette_effect))


# Thêm đường y = x
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Balanced (y=x)', alpha=0.7)

# Thêm mũi tên từ Original đến New Pipeline cho từng model family
for family in model_families_effect:
    original_data = df_pipeline_effect[(df_pipeline_effect['Model Family'] == family) & (df_pipeline_effect['Pipeline Type'] == 'Original')]
    new_pipeline_data = df_pipeline_effect[(df_pipeline_effect['Model Family'] == family) & (df_pipeline_effect['Pipeline Type'] == 'New Pipeline')]

    if not original_data.empty and not new_pipeline_data.empty:
        x_orig, y_orig = original_data['Label 0 (the same)'].iloc[0], original_data['Label 1 (the difference)'].iloc[0]
        x_new, y_new = new_pipeline_data['Label 0 (the same)'].iloc[0], new_pipeline_data['Label 1 (the difference)'].iloc[0]

        # Lấy màu của family đó từ color_map đã tạo
        color = color_map_effect[family]

        # Vẽ mũi tên
        plt.annotate('', xy=(x_new, y_new), xytext=(x_orig, y_orig),
                     arrowprops=dict(facecolor=color, shrink=0.05, width=1.5, headwidth=8, headlength=10, linewidth=1),
                     fontsize=9)

# Thêm chú thích cho từng điểm (Model Name) - Tinh chỉnh vị trí nhãn
for i, row in df_pipeline_effect.iterrows():
     label = row['Model'].replace('LLaVA One vision', 'LLaVA OV').replace('LLaVA Interleave', 'LLaVA IL').replace('Deepseek', 'DS').replace('Mantis', 'MT')
     label = label.replace('(new pipeline)', '(New)').replace('(direct return)', '(DR)').replace('Original', '(Orig)')
     x_pos, y_pos = row['Label 0 (the same)'], row['Label 1 (the difference)']

     # --- THAY ĐỔI: Logic đặt nhãn dựa trên vị trí ---
     # Đặt nhãn hơi xuống dưới và sang phải nếu L0 cao (gần mép phải)
     if x_pos > 0.9: # Ngưỡng L0 cao
         plt.text(x_pos + 0.015, y_pos - 0.01, label, fontsize=9, ha='left', va='top') # Đặt hơi xuống, căn theo đỉnh nhãn
     # Đặt nhãn hơi lên trên và sang phải nếu L0 thấp hơn
     else:
         plt.text(x_pos + 0.015, y_pos + 0.01, label, fontsize=9, ha='left', va='bottom') # Đặt hơi lên, căn theo đáy nhãn


plt.title('Effect of New Pipeline on Label 0 vs. Label 1 Performance')
plt.xlabel('Label 0 (the same) Score')
plt.ylabel('Label 1 (the difference) Score')
plt.xlim(0, 1.05) # Có thể nới rộng giới hạn trục nếu cần
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend(title='Model Family / Pipeline Type', loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_pipeline_effect_scatterplot.png'), dpi=300, bbox_inches='tight') # Lưu ảnh
plt.show()


# --- 4. Biểu đồ Cột Nhóm (Grouped Bar Chart) cho Macro F1 score (New Pipeline Insight Magnitude) ---

# Lọc dữ liệu cho các VLM mã nguồn mở và 3 loại pipeline
df_vlm_pipelines_macro = df_plottable[
    df_plottable['Model Family'].isin(['LLaVA One vision', 'LLaVA Interleave', 'Deepseek', 'Mantis'])
].copy()

plt.figure(figsize=(10, 6)) # Kích thước figure đơn
# Sử dụng palette màu custom đã định nghĩa để làm nổi bật New Pipeline
barplot_macro = sns.barplot(data=df_vlm_pipelines_macro,
                            x='Model Family', y='Macro F1 score', hue='Pipeline Type', palette=pipeline_colors)

plt.title('Macro F1 Score Comparison by Pipeline Type')
plt.xlabel('Model Family')
plt.ylabel('Macro F1 Score')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.xticks(rotation=0)
plt.legend(title='Pipeline Type', loc='upper left')

# Thêm giá trị lên trên các cột
for p in barplot_macro.patches:
    height = p.get_height()
    if not np.isnan(height): # Check for NaN values
        barplot_macro.annotate(f'{height:.3f}',
                               (p.get_x() + p.get_width() / 2., height),
                               ha='center', va='bottom',
                               xytext=(0, 3), # Offset text slightly above the bar
                               textcoords='offset points',
                               fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_macro_f1_grouped_bar_chart.png'), dpi=300, bbox_inches='tight') # Lưu ảnh
plt.show()


# --- 5. Biểu đồ Cột đơn (Bar Chart) xếp hạng các Model/Pipeline tốt nhất (Overall Ranking Insight) ---

# Chọn các dòng representing the best/key results for ranking
# Lấy bản 'New Pipeline' cho LLaVA, Deepseek, Mantis
# Lấy các dòng Facenet
# Bỏ qua ChatGPT 4o vì 'In proccess' và các dòng 'Original', 'Direct Return' không phải là kết quả cuối cùng so sánh tổng thể
df_ranking = df_plottable[
    (df_plottable['Pipeline Type'] == 'New Pipeline') |
    (df_plottable['Model Family'] == 'Facenet')
].copy()

# Tạo nhãn hiển thị trên biểu đồ
df_ranking['Plot Label'] = df_ranking['Model'].replace({
    'LLaVA One vision (new pipeline)': 'LLaVA OV (New)',
    'LLaVA Interleave (new pipeline)': 'LLaVA IL (New)',
    'Deepseek (new pipeline)': 'Deepseek (New)',
    'Mantis (new pipeline)': 'Mantis (New)',
    'Facenet: restnet_vggface': 'Facenet (VGG)', # Rút gọn nhãn Facenet
    'Facenet: restnet_webface': 'Facenet (Web)'
})

# Sắp xếp theo Macro F1 score giảm dần
df_ranking = df_ranking.sort_values('Macro F1 score', ascending=False)

plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=df_ranking, x='Plot Label', y='Macro F1 score', palette='viridis')

plt.title('Overall Macro F1 Score Ranking of Top Models/Pipelines')
plt.xlabel('Model / Pipeline')
plt.ylabel('Macro F1 Score')
plt.ylim(0, 1.05) # Đặt giới hạn trục Y để có khoảng trống cho nhãn
plt.xticks(rotation=45, ha='right') # Xoay nhãn trục X để dễ đọc

# Thêm giá trị lên trên mỗi cột
for p in barplot.patches:
    barplot.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)


plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_macro_f1_ranking_bar_chart.png'), dpi=300, bbox_inches='tight') # Lưu ảnh
plt.show()

# --- In bảng dữ liệu đầy đủ (để tham khảo) ---
print("\nBảng dữ liệu đầy đủ:")
print(df)