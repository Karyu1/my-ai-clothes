import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import zipfile
import time

# --- 核心图像处理函数 ---
def process_clothing(original_img, target_hue, s_weight, v_weight):
    # 1. AI 抠图获取主体蒙版
    no_bg_img = remove(original_img)
    no_bg_array = np.array(no_bg_img)
    subject_mask = no_bg_array[:, :, 3] 

    # 2. 转换 HSV 并识别皮肤
    rgb_img = np.array(original_img.convert('RGB'))
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    
    # 皮肤范围过滤 (HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv_img, lower_skin, upper_skin)
    
    # 3. 生成最终衣服蒙版 (主体区域 - 皮肤区域)
    clothes_mask = cv2.bitwise_and(subject_mask, cv2.bitwise_not(skin_mask))
    clothes_mask = cv2.GaussianBlur(clothes_mask, (7, 7), 0) # 平滑边缘

    # 4. 色彩转换逻辑
    hsv_float = hsv_img.astype(np.float32)
    h, s, v = cv2.split(hsv_float)
    
    h[:] = target_hue
    s = np.clip(s * s_weight, 0, 255)
    v = np.clip(v * v_weight, 0, 255)
    
    processed_rgb = cv2.cvtColor(cv2.merge((h, s, v)).astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # 5. Alpha 混合合成
    alpha = clothes_mask[:, :, np.newaxis] / 255.0
    final_img = (processed_rgb * alpha + rgb_img * (1 - alpha)).astype(np.uint8)
    
    return final_img

# --- Streamlit UI 界面 ---
st.set_page_config(page_title="服装批量换色 Pro", layout="wide")
st.title("👕 服装 AI 批量换色 & 自动打包系统")

with st.sidebar:
    st.header("🎨 调色配置")
    target_hue = st.select_slider(
        "选择目标颜色",
        options=list(range(0, 181, 10)),
        value=120,
        help="0:红, 30:橙, 60:黄, 90:绿, 120:蓝, 150:紫"
    )
    s_val = st.slider("饱和度 (色彩鲜艳度)", 0.5, 2.0, 1.0)
    v_val = st.slider("明亮度 (深浅度)", 0.5, 2.0, 1.0)
    st.divider()
    st.caption("注：本工具会自动保护肤色并移除背景影响。")

uploaded_files = st.file_uploader("上传图片 (支持批量)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    processed_results = []
    
    if st.button("开始批量处理"):
        progress_text = "AI 正在计算中，请稍候..."
        my_bar = st.progress(0, text=progress_text)
        
        cols = st.columns(3)
        
        for idx, file in enumerate(uploaded_files):
            # 运行处理逻辑
            img = Image.open(file)
            result_array = process_clothing(img, target_hue, s_val, v_val)
            
            # 存储结果供打包
            processed_results.append({"name": file.name, "img": result_array})
            
            # 实时预览
            with cols[idx % 3]:
                st.image(result_array, caption=f"预览: {file.name}", use_container_width=True)
            
            # 更新进度条
            progress = (idx + 1) / len(uploaded_files)
            my_bar.progress(progress, text=f"已完成 {idx+1}/{len(uploaded_files)}")
        
        st.success("✅ 所有图片处理完成！")
        
        # --- ZIP 打包下载逻辑 ---
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for item in processed_results:
                # 将 numpy 转为 JPEG 字节流
                res_pil = Image.fromarray(item["img"])
                img_io = io.BytesIO()
                res_pil.save(img_io, format='JPEG', quality=90)
                zip_file.writestr(f"colored_{item['name']}", img_io.getvalue())
        
        st.download_button(
            label="💾 点击下载全部处理后的图片 (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"batch_output_{int(time.time())}.zip",
            mime="application/zip",
            type="primary"
        )
