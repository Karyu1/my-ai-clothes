import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io, zipfile

# --- 1. 颜色中心采样（1:1 复刻参考色） ---
def get_ref_color_precise(ref_img):
    img = np.array(ref_img.convert('RGB'))
    h, w, _ = img.shape
    # 缩小采样窗口，只取中心颜色，防止采样到边框
    roi = img[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)]
    return np.mean(roi, axis=(0, 1))

# --- 2. 核心算法：语义保护 + 边缘净化 ---
def process_professional_transfer(original_img, target_rgb):
    # 图像预处理
    rgb_img = np.array(original_img.convert('RGB'))
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    
    # A. AI 主体剥离 (初步过滤背景)
    with st.spinner("正在定位服装区域..."):
        no_bg = remove(original_img, only_mask=True)
        subject_mask = np.array(no_bg)

    # B. 【精准肤色锁定】使用 YCrCb 排除人像
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    # 皮肤在 YCrCb 的典型分布范围
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    # 适当扩张皮肤掩模，确保边缘不漏色
    skin_mask = cv2.dilate(skin_mask, np.ones((7, 7), np.uint8), iterations=2)

    # C. 【道具与背景排除】
    # 利用原图的色彩饱和度和对比度，识别非衣物区域（如白色鞋子、灰色道具）
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY) # 排除纯白物体（如白鞋）
    _, black_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV) # 排除纯黑道具
    
    # D. 生成最终纯净服装掩模
    # 原理：主体区域 - 皮肤 - 白色物体 - 黑色物体
    clothes_mask = cv2.bitwise_and(subject_mask, cv2.bitwise_not(skin_mask))
    clothes_mask = cv2.bitwise_and(clothes_mask, cv2.bitwise_not(white_mask))
    clothes_mask = cv2.bitwise_and(clothes_mask, cv2.bitwise_not(black_mask))
    
    # 净化边缘：通过形态学处理消除“毛刺”和“杂色边”
    kernel = np.ones((5, 5), np.uint8)
    clothes_mask = cv2.morphologyEx(clothes_mask, cv2.MORPH_OPEN, kernel)
    clothes_mask_blur = cv2.GaussianBlur(clothes_mask, (11, 11), 0) / 255.0

    # E. 【高保真 1:1 换色逻辑】
    target_hsv = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # 强制克隆 H(色相) 和 S(饱和度)
    hsv_img[:, :, 0] = target_hsv[0]
    hsv_img[:, :, 1] = target_hsv[1]
    
    # 优化 V(明度) 通道：保留褶皱的同时，让暗部更有质感
    v = hsv_img[:, :, 2]
    v = cv2.normalize(v, None, alpha=max(50, target_hsv[2]-100), beta=min(255, target_hsv[2]+50), norm_type=cv2.NORM_MINMAX)
    hsv_img[:, :, 2] = v
    
    new_rgb = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # F. 最终无损合成
    m = clothes_mask_blur[:, :, np.newaxis]
    # 对衣物应用换色，其他部分（鞋子、脸、道具）100% 保持原样
    final = (new_rgb * m + rgb_img * (1 - m)).astype(np.uint8)
    return final

# --- 3. Streamlit 交互界面 ---
st.set_page_config(page_title="专业服装换色系统", layout="wide")
st.title("👔 专业级服装换色 (已修复道具/边缘杂色问题)")
st.markdown("---")

# 侧边栏：配置区
with st.sidebar:
    st.header("1. 参考颜色配置")
    ref_file = st.file_uploader("上传参考色图", type=['jpg', 'png', 'jpeg'])
    if ref_file:
        ref_img = Image.open(ref_file)
        st.image(ref_img, caption="参考色源")
        target_rgb = get_ref_color_precise(ref_img)
        st.success("颜色锁定成功")

# 主界面：上传与预览
st.header("2. 待换色照片上传")
uploaded_files = st.file_uploader("支持批量上传 (将自动排除鞋子/道具)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    with st.expander("🔍 预览已上传的原图"):
        cols = st.columns(5)
        for i, f in enumerate(uploaded_files):
            with cols[i % 5]:
                st.image(f, use_container_width=True)

if uploaded_files and ref_file:
    if st.button("🚀 开始精准换色处理"):
        t_rgb = get_ref_color_precise(Image.open(ref_file))
        res_cols = st.columns(2)
        zip_list = []
        
        for idx, file in enumerate(uploaded_files):
            try:
                res = process_professional_transfer(Image.open(file), t_rgb)
                zip_list.append({"name": file.name, "img": res})
                with res_cols[idx % 2]:
                    st.image(res, caption=f"处理结果: {file.name}", use_container_width=True)
            except Exception as e:
                st.error(f"处理出错: {file.name}")

        if zip_list:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as f:
                for item in zip_list:
                    img_io = io.BytesIO()
                    Image.fromarray(item["img"]).save(img_io, format='JPEG', quality=95)
                    f.writestr(f"fixed_{item['name']}", img_io.getvalue())
            st.download_button("💾 下载全部结果", buf.getvalue(), "output_fixed.zip")
else:
    st.warning("👈 请先确保已上传参考图和待处理照片。")
