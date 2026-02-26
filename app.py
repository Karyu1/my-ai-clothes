import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io, zipfile

# --- 1. 高精度参考色提取 ---
def get_ref_color_precise(ref_img):
    img = np.array(ref_img.convert('RGB'))
    h, w, _ = img.shape
    # 只取参考图正中心区域，避免背景颜色干扰
    roi = img[h//3:2*h//3, w//3:2*w//3]
    return np.mean(roi, axis=(0, 1))

# --- 2. 核心处理：1:1 颜色克隆与皮肤锁定 ---
def process_color_clone(original_img, target_rgb):
    # 转换为 RGB 数组
    rgb_img = np.array(original_img.convert('RGB'))
    
    # 步骤 A: AI 抠图（获取服装+人像主体）
    no_bg = remove(original_img)
    subject_alpha = np.array(no_bg)[:, :, 3]

    # 步骤 B: 深度皮肤锁定 (保护人脸、脖子、手部)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    # 覆盖更广泛的亚洲/欧洲人肤色范围
    lower_skin = np.array([0, 15, 30], dtype=np.uint8)
    upper_skin = np.array([28, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 对皮肤遮罩进行扩张，防止边缘“渗漏”绿色
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    # 步骤 C: 生成服装精准遮罩 (主体 - 皮肤)
    clothes_mask = cv2.bitwise_and(subject_alpha, cv2.bitwise_not(skin_mask))
    clothes_mask = cv2.GaussianBlur(clothes_mask, (11, 11), 0) / 255.0

    # 步骤 D: 1:1 质感克隆算法
    # 提取原图亮度 (L) 保持质感细节
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB).astype("float32")
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 将目标 RGB 转换为参考 Lab 色彩
    target_img_piece = np.full((1, 1, 3), target_rgb, dtype=np.uint8)
    target_lab = cv2.cvtColor(target_img_piece, cv2.COLOR_RGB2LAB)[0][0]
    
    # 只修改 a 和 b 颜色通道，L 通道（亮度/纹理）完全保持原样
    new_lab = cv2.merge([l_channel, np.full_like(a_channel, target_lab[1]), np.full_like(b_channel, target_lab[2])])
    new_rgb = cv2.cvtColor(new_lab.astype("uint8"), cv2.COLOR_LAB2RGB)

    # 步骤 E: 最终精准合成
    mask_3d = clothes_mask[:, :, np.newaxis]
    # 如果蒙版是 1，用新颜色；如果是 0，保留原图
    final_img = (new_rgb * mask_3d + rgb_img * (1 - mask_3d)).astype(np.uint8)
    
    return final_img

# --- 3. 网页界面设计 ---
st.set_page_config(page_title="1:1服装颜色克隆", layout="wide")
st.title("👕 AI 服装颜色 1:1 精准克隆系统")

# 侧边栏：参考图
with st.sidebar:
    st.header("🎨 参考色来源")
    ref_file = st.file_uploader("上传目标颜色参考图", type=['jpg', 'png', 'jpeg'])
    if ref_file:
        ref_img = Image.open(ref_file)
        st.image(ref_img, caption="参考图预览", use_container_width=True)
        t_rgb = get_ref_color_precise(ref_img)
        st.markdown(f"**已锁定目标 RGB:** `{int(t_rgb[0])}, {int(t_rgb[1])}, {int(t_rgb[2])}`")

# 主界面：待换色照片
st.header("📸 待换色照片管理")
uploaded_files = st.file_uploader("上传服装照片（支持多图，上传后可预览）", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

# 预览区
if uploaded_files:
    st.subheader("🖼️ 上传照片预览")
    pre_cols = st.columns(4)
    for i, file in enumerate(uploaded_files):
        with pre_cols[i % 4]:
            st.image(file, caption=file.name, use_container_width=True)

# 处理区
if uploaded_files and ref_file:
    if st.button("🚀 开始 AI 精准换色"):
        progress_bar = st.progress(0)
        output_images = []
        target_rgb = get_ref_color_precise(Image.open(ref_file))
        
        st.subheader("✨ 换色结果对比")
        res_cols = st.columns(2)
        
        for idx, file in enumerate(uploaded_files):
            try:
                res = process_color_clone(Image.open(file), target_rgb)
                output_images.append({"name": file.name, "img": res})
                
                with res_cols[idx % 2]:
                    st.image(res, caption=f"结果: {file.name}", use_container_width=True)
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            except Exception as e:
                st.error(f"处理 {file.name} 失败: {e}")

        # ZIP 下载功能
        if output_images:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for item in output_images:
                    img_io = io.BytesIO()
                    Image.fromarray(item["img"]).save(img_io, format='JPEG', quality=95)
                    zf.writestr(f"cloned_{item['name']}", img_io.getvalue())
            st.download_button("💾 下载全部处理好的图片 (ZIP)", zip_buf.getvalue(), "output_images.zip")
else:
    st.info("👈 请先在左侧上传参考图，然后在上方上传待换色的照片。")
