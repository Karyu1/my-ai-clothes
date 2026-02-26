import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io, zipfile

# --- 1. 核心颜色采样 ---
def get_ref_color_precise(ref_img):
    img = np.array(ref_img.convert('RGB'))
    h, w, _ = img.shape
    roi = img[h//3:2*h//3, w//3:2*w//3] # 取中心
    return np.mean(roi, axis=(0, 1))

# --- 2. 核心算法：语义保护 + 强力上色 ---
def process_final_v3(original_img, target_rgb):
    # 预处理
    rgb_img = np.array(original_img.convert('RGB'))
    h, w, _ = rgb_img.shape
    
    # A. 提取主体掩模
    mask = remove(original_img, only_mask=True)
    mask = np.array(mask)

    # B. 【多重禁区保护】
    # 1. 肤色保护 (YCrCb + HSV)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
    skin_hsv = cv2.inRange(hsv, (0, 15, 40), (25, 255, 255))
    skin_ycrcb = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    skin_mask = cv2.bitwise_or(skin_hsv, skin_ycrcb)
    skin_mask = cv2.dilate(skin_mask, np.ones((7,7), np.uint8), iterations=2)

    # 2. 地理位置保护 (保护鞋子：忽略图片底部 15% 的区域)
    geo_mask = np.ones((h, w), dtype=np.uint8) * 255
    geo_mask[int(h*0.88):, :] = 0 # 强制锁定底部

    # 3. 中性色保护 (保护白色鞋子、灰色道具)
    # 如果饱和度极低，说明是白/灰/黑，不应染上鲜艳颜色
    s_channel = hsv[:, :, 1]
    neutral_mask = cv2.threshold(s_channel, 25, 255, cv2.THRESH_BINARY_INV)[1]

    # C. 生成最终服装掩模 (主体 - 皮肤 - 底部 - 中性色)
    clothes_mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))
    clothes_mask = cv2.bitwise_and(clothes_mask, geo_mask)
    clothes_mask = cv2.bitwise_and(clothes_mask, cv2.bitwise_not(neutral_mask))
    
    # 柔化边缘，防止边缘出现锯齿和杂色
    clothes_mask = cv2.GaussianBlur(clothes_mask, (15, 15), 0) / 255.0

    # D. 【强力 1:1 上色逻辑】
    # 提取目标颜色的 LAB 特征
    target_img = np.full((1, 1, 3), target_rgb, dtype=np.uint8)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)[0][0]

    # 原图转 LAB
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab_img)

    # 针对深色衣服的亮度提升：让 L 通道向参考色靠近，而不是死黑
    l_target = target_lab[0]
    # 如果原图很暗，则大幅提升亮度以承载颜色
    l = np.where(l < 50, l * 1.5 + (l_target * 0.3), l)
    l = np.clip(l, 0, 255)

    # 强制克隆 A/B 颜色通道
    new_a = np.full_like(a, target_lab[1])
    new_b = np.full_like(b, target_lab[2])

    merged_lab = cv2.merge([l, new_a, new_b]).astype(np.uint8)
    new_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    # E. 合成结果
    m = clothes_mask[:, :, np.newaxis]
    final = (new_rgb * m + rgb_img * (1 - m)).astype(np.uint8)
    return final

# --- 3. UI 界面 ---
st.set_page_config(page_title="AI精准复刻", layout="wide")
st.title("👕 AI 服装颜色 1:1 深度复刻系统")
st.info("已启用：底部鞋子保护、灰白道具保护、深色衣服亮度增益。")

with st.sidebar:
    st.header("1. 目标参考色")
    ref_file = st.file_uploader("上传你想要复刻的颜色图", type=['jpg', 'png', 'jpeg'])
    if ref_file:
        ref_img = Image.open(ref_file)
        st.image(ref_img, caption="参考色源")
        t_rgb = get_ref_color_precise(ref_img)
        st.success("颜色已锁定")

st.header("2. 待处理照片")
uploaded_files = st.file_uploader("支持批量上传（上传后下方自动预览）", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    with st.expander("👁️ 待处理原图预览", expanded=True):
        p_cols = st.columns(6)
        for i, f in enumerate(uploaded_files):
            p_cols[i % 6].image(f, use_container_width=True)

if uploaded_files and ref_file:
    if st.button("🚀 开始 1:1 复刻上色"):
        t_rgb = get_ref_color_precise(Image.open(ref_file))
        res_cols = st.columns(2)
        results = []
        
        for idx, file in enumerate(uploaded_files):
            try:
                res = process_final_v3(Image.open(file), t_rgb)
                results.append({"name": file.name, "img": res})
                res_cols[idx % 2].image(res, caption=f"复刻成功: {file.name}", use_container_width=True)
            except Exception as e:
                st.error(f"处理失败: {file.name}")

        if results:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as f:
                for r in results:
                    img_io = io.BytesIO()
                    Image.fromarray(r["img"]).save(img_io, format='JPEG', quality=95)
                    f.writestr(f"final_{r['name']}", img_io.getvalue())
            st.download_button("💾 下载所有处理结果", zip_buf.getvalue(), "cloned_results.zip")
