import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import zipfile
import time

# --- 1. 自动提取参考图颜色 ---
def extract_target_color(reference_img):
    img = np.array(reference_img.convert('RGB'))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, w, _ = img_hsv.shape
    roi = img_hsv[h//3:2*h//3, w//3:2*w//3]
    avg_hsv = cv2.mean(roi)
    return avg_hsv[0], avg_hsv[1], avg_hsv[2] # 返回 H, S, V

# --- 2. 核心处理函数：支持黑色服装变色 ---
def process_advanced_color(original_img, t_h, t_s, t_v, s_weight, v_weight):
    # AI 自动扣图
    no_bg_img = remove(original_img)
    no_bg_array = np.array(no_bg_img)
    subject_mask = no_bg_array[:, :, 3] 

    # 转换原图到 HSV
    rgb_img = np.array(original_img.convert('RGB'))
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv_img)
    
    # --- 肤色保护 ---
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    hsv_uint8 = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    skin_mask = cv2.inRange(hsv_uint8, lower_skin, upper_skin)
    clothes_mask = cv2.bitwise_and(subject_mask, cv2.bitwise_not(skin_mask))
    clothes_mask = cv2.GaussianBlur(clothes_mask, (7, 7), 0) / 255.0

    # --- 针对黑色衣服的特殊算法 ---
    # 1. 提取目标色相
    h[:] = t_h
    
    # 2. 强制提升饱和度 (解决黑色无色问题)
    # 如果原图饱和度低，则使用参考图的饱和度乘以权重
    s = np.where(s < 50, t_s * s_weight, s * s_weight)
    s = np.clip(s, 0, 255)
    
    # 3. 提升明亮度 (解决黑色太暗无法上色问题)
    # 黑色衣服如果不提亮，颜色是染不上去的。我们保留纹理的同时拉高亮度。
    v_boost = np.where(v < 100, v + (255 - v) * 0.4 * v_weight, v * v_weight)
    v = np.clip(v_boost, 0, 255)
    
    processed_hsv = cv2.merge((h, s, v)).astype(np.uint8)
    processed_rgb = cv2.cvtColor(processed_hsv, cv2.COLOR_HSV2RGB)
    
    # --- 最终合成 ---
    alpha = clothes_mask[:, :, np.newaxis]
    final_img = (processed_rgb * alpha + rgb_img * (1 - alpha)).astype(np.uint8)
    return final_img

# --- 3. UI 界面 ---
st.set_page_config(page_title="黑色服装专项换色", layout="wide")
st.title("👕 AI 服装换色 (黑色/深色服装专项版)")

with st.sidebar:
    st.header("1️⃣ 参考图上传")
    ref_file = st.file_uploader("上传目标颜色参考图", type=['jpg', 'png', 'jpeg'])
    
    t_h, t_s, t_v = 120, 150, 150 # 默认值
    if ref_file:
        ref_img = Image.open(ref_file)
        t_h, t_s, t_v = extract_target_color(ref_img)
        st.image(ref_img, caption="提取颜色成功", width=150)

    st.header("2️⃣ 黑色服装微调")
    st.write("如果是黑色衣服，请调大下方两个参数：")
    s_weight = st.slider("饱和度补偿", 0.0, 3.0, 1.5)
    v_weight = st.slider("明亮度补偿", 0.0, 3.0, 1.2)

st.subheader("3️⃣ 上传需要换色的照片")
uploaded_files = st.file_uploader("支持批量上传", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 开始 AI 换色"):
        cols = st.columns(2)
        for idx, file in enumerate(uploaded_files):
            res = process_advanced_color(Image.open(file), t_h, t_s, t_v, s_weight, v_weight)
            with cols[idx % 2]:
                st.image(res, caption=f"处理结果: {file.name}", use_container_width=True)
