import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io, zipfile

# --- 1. 获取参考图核心颜色 ---
def get_ref_color(ref_img):
    img = np.array(ref_img.convert('RGB'))
    h, w, _ = img.shape
    # 取中心一小块，避开背景干扰
    roi = img[h//3:2*h//3, w//3:2*w//3]
    return np.mean(roi, axis=(0, 1))

# --- 2. 核心处理逻辑 ---
def process_perfect(original_img, target_rgb):
    # 抠图获取人像
    no_bg = remove(original_img)
    subject_alpha = np.array(no_bg)[:, :, 3]

    # 原图转为 RGB 和 HSV
    orig_rgb = np.array(original_img.convert('RGB'))
    hsv = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2HSV)

    # 【肤色保护】识别范围：涵盖各种肤色阴影
    lower_skin = np.array([0, 10, 40], dtype=np.uint8)
    upper_skin = np.array([35, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 肤色区域向外稍微扩张，防止边缘“渗色”
    skin_mask = cv2.dilate(skin_mask, np.ones((5, 5), np.uint8), iterations=1)

    # 最终服装蒙版 = (主体 - 肤色)
    clothes_mask = cv2.bitwise_and(subject_alpha, cv2.bitwise_not(skin_mask))
    clothes_mask = cv2.GaussianBlur(clothes_mask, (7, 7), 0) / 255.0

    # 【质感保留染色】
    # 将目标颜色铺满全图
    color_layer = np.full(orig_rgb.shape, target_rgb, dtype=np.float32)
    
    # 获取原图的明度细节（褶皱）
    gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # 染色逻辑：颜色 * 亮度引导 + 针对深色的补偿
    # 这能保证黑色衣服变色，同时不丢失褶皱
    res_layer = color_layer * (gray[:, :, np.newaxis] * 0.8 + 0.2)
    res_layer = np.clip(res_layer, 0, 255).astype(np.uint8)

    # 合成：人脸和背景用原图，服装用染色图
    mask_3d = clothes_mask[:, :, np.newaxis]
    final_img = (res_layer * mask_3d + orig_rgb * (1 - mask_3d)).astype(np.uint8)
    return final_img

# --- 3. 界面设计 ---
st.set_page_config(page_title="AI服装变色极速版", layout="wide")
st.title("👕 AI 智能服装换色（终极版）")

with st.sidebar:
    st.header("1. 参考颜色")
    ref_file = st.file_uploader("上传色卡/参考图", type=['jpg', 'png', 'jpeg'])
    
st.header("2. 批量处理")
uploaded_files = st.file_uploader("上传待换色照片", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files and ref_file:
    if st.button("🚀 开始精准换色"):
        target_rgb = get_ref_color(Image.open(ref_file))
        cols = st.columns(2)
        results = []
        
        for idx, file in enumerate(uploaded_files):
            try:
                res = process_perfect(Image.open(file), target_rgb)
                results.append({"name": file.name, "img": res})
                with cols[idx % 2]:
                    st.image(res, caption=f"已完成: {file.name}", use_container_width=True)
            except Exception as e:
                st.error(f"处理 {file.name} 出错: {e}")

        if results:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as f:
                for r in results:
                    img_buf = io.BytesIO()
                    Image.fromarray(r["img"]).save(img_buf, format='JPEG')
                    f.writestr(f"new_{r['name']}", img_buf.getvalue())
            st.download_button("📦 下载 ZIP 压缩包", buf.getvalue(), "output.zip")
else:
    st.info("请先上传参考颜色图，再上传需要处理的照片。")
