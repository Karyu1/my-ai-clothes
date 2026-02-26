import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io, zipfile

# --- 1. 高精度采样 ---
def get_ref_color_precise(ref_img):
    img = np.array(ref_img.convert('RGB'))
    h, w, _ = img.shape
    roi = img[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)]
    return np.mean(roi, axis=(0, 1))

# --- 2. 核心处理逻辑 ---
def process_ultimate(original_img, target_rgb):
    # 转换色彩空间
    rgb_img = np.array(original_img.convert('RGB'))
    h, w, _ = rgb_img.shape
    
    # A. 提取主体 (使用 AI 抠图)
    mask = remove(original_img, only_mask=True)
    mask = np.array(mask)

    # B. 【多重保护防御系统】
    # 1. 深度皮肤防御 (YCrCb 空间)
    ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    skin_mask = cv2.dilate(skin_mask, np.ones((5,5), np.uint8), iterations=2)

    # 2. 地理位置防御 (强制排除底部 12% 区域，保护鞋子)
    geo_mask = np.ones((h, w), dtype=np.uint8) * 255
    geo_mask[int(h*0.88):, :] = 0 

    # 3. 中性色防御 (识别白色道具、浅灰色背景)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]
    neutral_mask = cv2.threshold(s_channel, 35, 255, cv2.THRESH_BINARY_INV)[1]

    # C. 合成最终衣物掩模
    clothes_mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))
    clothes_mask = cv2.bitwise_and(clothes_mask, geo_mask)
    clothes_mask = cv2.bitwise_and(clothes_mask, cv2.bitwise_not(neutral_mask))
    clothes_mask = cv2.GaussianBlur(clothes_mask, (11, 11), 0) / 255.0

    # D. 【强力 1:1 色彩克隆】
    target_img = np.full((1, 1, 3), target_rgb, dtype=np.uint8)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)[0][0]

    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab_img)

    # 关键：针对黑色衣服提升亮度(L)
    # 如果原图亮度低于阈值，则向目标亮度靠拢，使其能上色
    l_target = target_lab[0]
    l = np.where(l < 60, l * 0.5 + (l_target * 0.7), l) 
    l = np.clip(l, 0, 255)

    # 强制覆盖颜色通道 (a, b)
    new_a = np.full_like(a, target_lab[1])
    new_b = np.full_like(b, target_lab[2])

    merged_lab = cv2.merge([l, new_a, new_b]).astype(np.uint8)
    new_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    # E. 最终融合
    m = clothes_mask[:, :, np.newaxis]
    final = (new_rgb * m + rgb_img * (1 - m)).astype(np.uint8)
    return final

# --- 3. UI 界面 ---
st.set_page_config(page_title="AI精准变色系统", layout="wide")
st.title("👕 AI 1:1 颜色深度克隆 (已解决鞋子/道具染色问题)")

with st.sidebar:
    st.header("1. 颜色参考")
    ref_file = st.file_uploader("上传目标颜色图", type=['jpg', 'png', 'jpeg'])
    if ref_file:
        ref_img = Image.open(ref_file)
        st.image(ref_img, caption="已提取颜色", width=200)
        t_rgb = get_ref_color_precise(ref_img)

st.header("2. 待处理照片")
uploaded_files = st.file_uploader("可批量上传原图", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    with st.expander("🔍 预览已上传的照片", expanded=True):
        cols = st.columns(4)
        for i, f in enumerate(uploaded_files):
            cols[i % 4].image(f, use_container_width=True)

if uploaded_files and ref_file:
    if st.button("🚀 执行精准换色"):
        t_rgb = get_ref_color_precise(Image.open(ref_file))
        res_cols = st.columns(2)
        results = []
        
        for idx, file in enumerate(uploaded_files):
            try:
                # 核心处理
                res = process_ultimate(Image.open(file), t_rgb)
                results.append({"name": file.name, "img": res})
                # 展示预览
                res_cols[idx % 2].image(res, caption=f"结果: {file.name}", use_container_width=True)
            except Exception as e:
                st.error(f"处理 {file.name} 时出错")

        if results:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as f:
                for r in results:
                    img_buf = io.BytesIO()
                    Image.fromarray(r["img"]).save(img_buf, format='JPEG', quality=95)
                    f.writestr(f"fixed_{r['name']}", img_buf.getvalue())
            st.download_button("💾 下载所有处理结果", buf.getvalue(), "output.zip")
else:
    st.info("👈 请先上传左侧颜色参考图，再上传需要变色的照片。")
