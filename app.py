import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io, zipfile

# --- 修复版色彩转换函数 ---
def color_transfer(source, target):
    # 强制转换为 float32 且确保只有 3 通道
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # 计算均值和标准差
    (l_src, a_src, b_src) = cv2.split(source_lab)
    (l_tar, a_tar, b_tar) = cv2.split(target_lab)

    def scale_channel(src, tar):
        s_mean, s_std = cv2.meanStdDev(src)
        t_mean, t_std = cv2.meanStdDev(tar)
        # 核心迁移公式
        res = ((tar - t_mean) * (s_std / (t_std + 1e-5))) + s_mean
        return np.clip(res, 0, 255)

    l_new = scale_channel(l_src, l_tar)
    a_new = scale_channel(a_src, a_tar)
    b_new = scale_channel(b_src, b_tar)

    transfer = cv2.merge([l_new, a_new, b_new]).astype("uint8")
    return cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)

def process_auto_match(original_img, ref_img):
    # 【关键修复点 1】: 强制 convert('RGB') 剥离透明通道
    ref_rgb = np.array(ref_img.convert('RGB'))
    ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)

    # 1. AI 抠图
    no_bg = remove(original_img)
    mask = np.array(no_bg)[:, :, 3]
    
    # 【关键修复点 2】: 待换色图也强制 RGB 处理
    rgb_img = np.array(original_img.convert('RGB'))
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # 2. 肤色检测
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 3. 混合蒙版
    clothes_mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))
    clothes_mask = cv2.GaussianBlur(clothes_mask, (7, 7), 0) / 255.0

    # 4. 执行颜色克隆
    matched_bgr = color_transfer(ref_bgr, bgr_img)
    matched_rgb = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)

    # 5. 合成
    alpha = clothes_mask[:, :, np.newaxis]
    final = (matched_rgb * alpha + rgb_img * (1 - alpha)).astype(np.uint8)
    return final

# --- 简化 UI ---
st.set_page_config(page_title="AI全自动换色", layout="wide")
st.title("👕 修复版：AI 颜色细节克隆")

with st.sidebar:
    st.header("1. 参考颜色图")
    ref_file = st.file_uploader("上传目标颜色参考", type=['jpg', 'png', 'jpeg'])
    if ref_file:
        st.image(ref_file)

st.header("2. 待换色服装图")
uploaded_files = st.file_uploader("支持批量上传", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files and ref_file:
    if st.button("🚀 开始处理"):
        ref_img = Image.open(ref_file)
        cols = st.columns(2)
        for idx, file in enumerate(uploaded_files):
            try:
                res = process_auto_match(Image.open(file), ref_img)
                with cols[idx % 2]:
                    st.image(res, use_container_width=True)
            except Exception as e:
                st.error(f"处理图片 {file.name} 时出错：{e}")
