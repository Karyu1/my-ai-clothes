import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import zipfile

# --- 核心算法：Lab空间色彩转换 (自动匹配，无需微调) ---
def color_transfer(source, target):
    # 将图片转为 Lab 空间（更能模拟人类视觉，对黑白变色更友好）
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # 计算参考图(source)和原图(target)的均值和标准差
    (l_mean_src, l_std_src, a_mean_src, a_std_src, b_mean_src, b_std_src) = cv2.meanStdDev(source_lab)
    (l_mean_tar, l_std_tar, a_mean_tar, a_std_tar, b_mean_tar, b_std_tar) = cv2.meanStdDev(target_lab)

    # 分离通道
    (l, a, b) = cv2.split(target_lab)

    # 执行颜色迁移：让原图的分布贴近参考图
    l = ((l - l_mean_tar) * (l_std_src / (l_std_tar + 1e-5))) + l_mean_src
    a = ((a - a_mean_tar) * (a_std_src / (a_std_tar + 1e-5))) + a_mean_src
    b = ((b - b_mean_tar) * (b_std_src / (b_std_tar + 1e-5))) + b_mean_src

    # 裁剪范围并转换回 BGR
    transfer = cv2.merge([l, a, b])
    transfer = np.clip(transfer, 0, 255).astype("uint8")
    transfer = cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)
    return transfer

def process_auto_match(original_img, ref_img):
    # 1. AI 抠图
    no_bg = remove(original_img)
    mask = np.array(no_bg)[:, :, 3]
    rgb_img = np.array(original_img.convert('RGB'))
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # 2. 准备参考图
    ref_bgr = cv2.cvtColor(np.array(ref_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # 3. 肤色检测 (保护脸部细节)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 衣服蒙版 = 主体 - 皮肤
    clothes_mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))
    clothes_mask = cv2.GaussianBlur(clothes_mask, (5, 5), 0) / 255.0

    # 4. 执行颜色自动克隆
    matched_bgr = color_transfer(ref_bgr, bgr_img)
    matched_rgb = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)

    # 5. 合成
    alpha = clothes_mask[:, :, np.newaxis]
    final = (matched_rgb * alpha + rgb_img * (1 - alpha)).astype(np.uint8)
    return final

# --- 简化版网页界面 ---
st.set_page_config(page_title="AI全自动换色器", layout="wide")
st.title("👕 AI 全自动服装颜色克隆")
st.markdown("只需上传参考图，系统将自动匹配颜色与质感，无需手动调节。")

with st.sidebar:
    st.header("1. 上传颜色参考图")
    ref_file = st.file_uploader("此图颜色将作为目标色", type=['jpg', 'png', 'jpeg'])
    if ref_file:
        st.image(ref_file, caption="目标颜色参考", use_container_width=True)

st.header("2. 上传待换色照片")
uploaded_files = st.file_uploader("支持批量上传", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files and ref_file:
    if st.button("🚀 开始全自动克隆颜色"):
        ref_img = Image.open(ref_file)
        cols = st.columns(2)
        for idx, file in enumerate(uploaded_files):
            res = process_auto_match(Image.open(file), ref_img)
            with cols[idx % 2]:
                st.image(res, caption=f"自动匹配结果: {file.name}", use_container_width=True)
else:
    st.warning("请确保同时上传了【参考图】和【待换色图】")
