import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import zipfile

# --- 1. 获取参考图最纯正的颜色 ---
def get_target_color(ref_img):
    img = np.array(ref_img.convert('RGB'))
    # 取中心 50x50 像素的平均值，避开边缘背景
    h, w, _ = img.shape
    roi = img[h//2-25:h//2+25, w//2-25:w//2+25]
    avg_color = np.mean(roi, axis=(0, 1))
    return avg_color # 返回 [R, G, B]

# --- 2. 核心处理：精准保护与质感染色 ---
def process_high_precision(original_img, target_rgb):
    # 第一步：AI 抠图获取人像主体
    no_bg = remove(original_img)
    subject_mask = np.array(no_bg)[:, :, 3] # 提取 Alpha 通道

    # 第二步：肤色精准识别 (扩大范围防止漏掉脸部和脖子)
    rgb_img = np.array(original_img.convert('RGB'))
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    
    # 更宽的肤色检测范围：涵盖偏黄、偏红和阴影下的皮肤
    lower_skin = np.array([0, 15, 40], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv_img, lower_skin, upper_skin)
    
    # 细节优化：对皮肤遮罩进行膨胀，确保边缘不留色边
    kernel = np.ones((5,5), np.uint8)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

    # 第三步：生成最终衣服遮罩 (主体 - 皮肤)
    clothes_mask = cv2.bitwise_and(subject_mask, cv2.bitwise_not(skin_mask))
    clothes_mask = cv2.GaussianBlur(clothes_mask, (7, 7), 0) / 255.0

    # 第四步：质感保留变色算法
    # 将原图转为灰度，以此提取明暗细节（褶皱、光影）
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # 创建目标颜色层
    color_layer = np.full(rgb_img.shape, target_rgb, dtype=np.float32)
    
    # 使用“正片叠底”或“柔光”逻辑融合，确保质感不变
    # 这里使用灰度图作为亮度引导
    res_layer = color_layer * gray[:, :, np.newaxis]
    
    # 针对深色/黑色衣服进行亮度补偿
    res_layer = np.clip(res_layer * 1.2, 0, 255).astype(np.uint8)

    # 第五步：合成结果
    alpha = clothes_mask[:, :, np.newaxis]
    final_img = (res_layer * alpha + rgb_img * (1 - alpha)).astype(np.uint8)
    return final_img

# --- 3. 网页界面 ---
st.set_page_config(page_title="高精度AI换色", layout="wide")
st.title("👕 高精度 AI 服装换色系统")
st.info("本版本优化了肤色保护机制，确保脸部不变色，并精准还原参考图色彩。")

with st.sidebar:
    st.header("1. 上传颜色参考图")
    ref_file = st.file_uploader("请上传色卡或颜色样板", type=['jpg', 'png', 'jpeg'])
    if ref_file:
        ref_img = Image.open(ref_file)
        target_rgb = get_target_color(ref_img)
        st.image(ref_img, caption="参考颜色源")
        st.markdown(f"已锁定颜色: <div style='width:50px;height:20px;background-color:rgb({int(target_rgb[0])},{int(target_rgb[1])},{int(target_rgb[2])});display:inline-block;vertical-align:middle;'></div>", unsafe_allow_html=True)

st.header("2. 上传待处理照片")
uploaded_files = st.file_uploader("支持批量上传", type=['jpg
