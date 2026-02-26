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
    # 将 PIL 转为 OpenCV 格式
    img = np.array(reference_img.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # 取图片中心区域的平均颜色，避免边缘背景干扰
    h, w, _ = img.shape
    roi = img[h//3:2*h//3, w//3:2*w//3]
    avg_hsv = cv2.mean(roi)
    return avg_hsv[0] # 返回提取到的 H (色相) 值

# --- 2. 核心处理函数：换色 + 质感保留 + 肤色保护 ---
def process_advanced_color(original_img, target_hue, s_weight, v_weight):
    # AI 自动扣图 (获取主体 Mask)
    no_bg_img = remove(original_img)
    no_bg_array = np.array(no_bg_img)
    subject_mask = no_bg_array[:, :, 3] 

    # 转换原图到 HSV
    rgb_img = np.array(original_img.convert('RGB'))
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    
    # --- 肤色保护逻辑 ---
    # 定义典型的肤色 HSV 范围
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv_img, lower_skin, upper_skin)
    
    # 衣服区域 = (扣图主体) 排除 (皮肤区域)
    clothes_mask = cv2.bitwise_and(subject_mask, cv2.bitwise_not(skin_mask))
    # 羽化蒙版边缘，让颜色过渡更自然
    clothes_mask = cv2.GaussianBlur(clothes_mask, (7, 7), 0)

    # --- 质感保留变色 ---
    hsv_float = hsv_img.astype(np.float32)
    h, s, v = cv2.split(hsv_float)
    
    # 关键：只替换 H (色相)，保留原图的 V (亮度/质感)
    h[:] = target_hue
    s = np.clip(s * s_weight, 0, 255) # 调节饱和度
    v = np.clip(v * v_weight, 0, 255) # 调节明暗
    
    processed_hsv = cv2.merge((h, s, v)).astype(np.uint8)
    processed_rgb = cv2.cvtColor(processed_hsv, cv2.COLOR_HSV2RGB)
    
    # --- 最终合成 ---
    alpha = clothes_mask[:, :, np.newaxis] / 255.0
    # 结果 = 新颜色图 * 衣服蒙版 + 原图 * (1 - 衣服蒙版)
    final_img = (processed_rgb * alpha + rgb_img * (1 - alpha)).astype(np.uint8)
    return final_img

# --- 3. Streamlit 网页界面 ---
st.set_page_config(page_title="高级服装换色系统", layout="wide")
st.title("👕 AI 服装智能换色 (参考图取色版)")

with st.sidebar:
    st.header("1️⃣ 第一步：参考色提取")
    ref_file = st.file_uploader("上传参考图/色卡", type=['jpg', 'png', 'jpeg'])
    
    target_h = 120 # 默认蓝色
    if ref_file:
        ref_img = Image.open(ref_file)
        target_h = extract_target_color(ref_img)
        st.image(ref_img, caption="已提取此图颜色", width=150)
        st.success(f"已自动匹配色调: {int(target_h)}")

    st.header("2️⃣ 第二步：参数微调")
    s_val = st.slider("饱和度 (颜色浓淡)", 0.0, 2.0, 1.0)
    v_val = st.slider("明亮度 (深浅质感)", 0.0, 2.0, 1.0)
    st.info("💡 提示：即便上传了参考图，你依然可以微调颜色深浅。")

# 主界面：图片上传
st.subheader("3️⃣ 第三步：上传需要变色的服装照片")
uploaded_files = st.file_uploader("支持批量上传", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 开始 AI 换色处理"):
        processed_images = []
        cols = st.columns(2)
        progress = st.progress(0)
        
        for idx, file in enumerate(uploaded_files):
            # 处理
            img = Image.open(file)
            res = process_advanced_color(img, target_h, s_val, v_val)
            processed_images.append({"name": file.name, "img": res})
            
            # 显示
            with cols[idx % 2]:
                st.image(res, caption=f"处理结果: {file.name}", use_container_width=True)
            progress.progress((idx + 1) / len(uploaded_files))
            
        # 批量打包下载
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for item in processed_images:
                img_io = io.BytesIO()
                Image.fromarray(item["img"]).save(img_io, format='JPEG', quality=95)
                zf.writestr(f"new_{item['name']}", img_io.getvalue())
        
        st.download_button(
            label="📦 点击下载全部处理好的图片 (ZIP)",
            data=zip_buf.getvalue(),
            file_name="clothes_results.zip",
            mime="application/zip"
        )
