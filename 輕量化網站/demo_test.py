import streamlit as st #互動式介面化套件
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas #自訂輸入圖片套件
import os
import json
from HopfieldNetwork import HopfieldNetwork
from PatternGenerator import PatternGenerator

def plot_pattern(pattern, title='Pattern'): #將圖片顯示在互動式介面上
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(pattern.reshape(16, 16), cmap='gray', interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    st.pyplot(fig, use_container_width=True)

def plot_heatmap(matrix, title): #將遮罩圖像化
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    cax = ax.matshow(matrix, cmap='coolwarm')
    ax.set_title(title, fontsize=10)
    plt.colorbar(cax, fraction=0.046, pad=0.04)
    st.pyplot(fig, use_container_width=True)

def draw_custom_pattern(): #讓使用者可以自訂輸入圖片
    st.write("### 自定義圖形")
    canvas = st_canvas(
        fill_color="white",
        stroke_color="black",
        stroke_width=40,
        background_color="white",
        width=320,
        height=320,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas.image_data is not None:
        rgba_array = np.array(canvas.image_data).astype(np.uint8)
        rgb_array = rgba_array[:, :, :3]
        img = Image.fromarray(rgb_array)
        img_gray = img.convert("L")
        img_resized = img_gray.resize((16, 16), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        binary_pattern = np.where(img_array > 128, 1, -1)
        return binary_pattern.flatten()
    return None

def load_patterns(folder="saved_patterns", n=100):
        with open(os.path.join(folder, "metadata.json"), "r") as f:
            metadata = json.load(f)

        selected = list(metadata.items())[:n]
        patterns = {}
        for filename, info in selected:
            pattern = np.load(os.path.join(folder, filename))
            key_name = f"自動圖形 {filename.split('.')[0][-3:]}"
            patterns[key_name] = pattern
        return patterns

# --- Streamlit UI --- 以下全都是互動式介面設定
st.set_page_config(layout="wide")
st.title("Hopfield Network Demo with Memory Impairment")

cols = st.columns([1, 1, 1])
with cols[0]:
    st.subheader("選擇圖形")
    pattern_options = {
        "全白": np.ones(256),
        "全黑": -np.ones(256),
        "對角線": np.array([1 if i//16 == i%16 else -1 for i in range(256)]),
        "十字": np.array([1 if (i%16 == 8 or i//16 == 8) else -1 for i in range(256)]),
        "方塊": np.array([1 if (4<=i//16<=11 and 4<=i%16<=11) else -1 for i in range(256)]),
        "自訂": "custom",
    }
    patterns = load_patterns()
    pattern_options.update(patterns)  # 將自動生成的圖形加進選單

    generator = PatternGenerator()

    selected_patterns = st.multiselect("選擇要記住的圖形（可多選）", list(pattern_options.keys()), default=["十字"])
    complexity = generator.normalized_complexity(pattern_options["十字"])
    pattern_list = []
    for name in selected_patterns:
        if name == "自訂":
            custom = draw_custom_pattern()
            if custom is not None:
                complexity = generator.normalized_complexity(custom)
                pattern_list.append(custom)
        else:
            complexity = generator.normalized_complexity(pattern_options[name])
            pattern_list.append(pattern_options[name])

    st.metric("圖形複雜度", f"{complexity:.2f}")

with cols[1]: #可調整的參數滑桿
    st.subheader("網路設定")
    death_rate = st.slider("突觸功能衰弱比例（突觸稀疏）", 0.0, 1.0, 0.3, 0.01)
    residual_strength = st.slider("突觸功能衰弱強度", 0.0, 1.0, 0.4, 0.01)
    plaque_intensity = st.slider("斑塊影響強度（神經元功能降低）", 0.0, 1.0, 0.4, 0.05)
    plaque_count = st.slider("斑塊數量", 1, 10, 3, 1)
    noise_level = st.slider("輸入噪聲強度", 0.0, 0.5, 0.2, 0.05)

with cols[2]:
    st.subheader("測試輸入")
    test_pattern_name = st.selectbox("選擇要回憶的圖形", selected_patterns)
    if test_pattern_name == "自訂":
        input_pattern = pattern_list[selected_patterns.index("自訂")]
    else:
        input_pattern = pattern_options[test_pattern_name]

hop = HopfieldNetwork(death_rate=death_rate, residual_strength=residual_strength, plaque_intensity=plaque_intensity, 
                      plaque_count=plaque_count)
hop.train(pattern_list)

noisy_input = hop.add_noise(input_pattern, noise_level=noise_level)
recalled = hop.recall(noisy_input)

st.subheader("記憶與回憶視覺化")
vcols = st.columns(3)
with vcols[0]:
    plot_pattern(input_pattern, "Input origin pattern")
with vcols[1]:
    plot_pattern(noisy_input, "Pattern with noise")
with vcols[2]:
    plot_pattern(recalled, "Pattern after recall")

success_rate = np.mean(recalled == input_pattern)
st.metric("回憶成功率", f"{success_rate*100:.2f}%")

st.subheader("權重損傷視覺化(1表示健康)")
hcols = st.columns(2)
with hcols[0]:
    plot_heatmap(hop.death_mask, "Synapse sparse distribution")
with hcols[1]:
    plot_heatmap(hop.plaque_mask, "Plaque distribution")


