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
    # 采样中心 100x100 区域避免背景色干扰
    roi = img[h//2-50:h//2+50, w//2-50:w//2+100]
    return np.mean(roi, axis=(0, 1))

# --- 2. 核心算法：高保元换色 ---
def process_perfect_match(original_img, target_rgb):
    # 转为数组
    rgb_img = np.array(original_img.convert('RGB'))
    
    # A. 剥离背景
    no_bg = remove(original_img)
    subject_mask = np.array(no_bg)[:, :, 3]

    # B. 【人脸/皮肤绝对防御】
    # 结合 HSV 和 YCrCb 两种空间识别人脸，防止变色
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
    
    # HSV 范围
    lower_hsv = np.array([0, 10, 40], dtype=np.uint8)
    upper_hsv = np.array([30, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # YCrCb 范围 (识别人脸精准度极高)
    mask_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    
    # 合并皮肤掩模并向外大幅膨胀，确保脖子/发际线边缘不留绿边
    skin_mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    skin_mask = cv2.dilate(skin_mask, np.ones((9, 9), np.uint8), iterations=2)

    # C. 生成服装纯净掩模
    clothes_mask = cv2.bitwise_and(subject_mask, cv2.bitwise_not(skin_mask))
    clothes_mask = cv2.GaussianBlur(clothes_mask, (15, 15), 0) / 255.0

    # D. 【1:1 颜色克隆算法】
    # 使用 HSV 偏移 + 亮度对齐，保留 1:1 的颜色和褶皱细节
    target_hsv = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    
    # 转换原图到浮点 HSV
    hsv_float = hsv.astype(np.float32)
    
    # 强制将 H(色相) 和 S(饱和度) 设置为参考色
    hsv_float[:, :, 0] = target_hsv[0] # H
    hsv_float[:, :, 1] = target_hsv[1] # S
    
    # 亮度(V) 通道特殊处理：保留原图纹理，但提升黑色衣服的整体亮度
    v_chan = hsv_float[:, :, 2]
    # 对亮度进行非线性提升，让颜色更通透不发灰
    v_chan = np.where(v_chan < 128, v_chan * 1.2, v_chan)
    hsv_float[:, :, 2] = np.clip(v_chan, 0, 255)
    
    new_rgb = cv2.cvtColor(hsv_float.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # E. 最终合成
    m = clothes_mask[:, :, np.newaxis]
    # 对衣服区域应用新颜色，其余完全保留原图
    final = (new_rgb * m + rgb_img * (1 - m)).astype(np.uint8)
    return final

# --- 3. UI 界面 ---
st.set_page_config(page_title="AI精准服装换色", layout="wide")
st.title("👕 AI 1:1 颜色克隆 (专业修复版)")

with st.sidebar:
    st.header("1. 参考颜色图")
    ref_file = st.file_uploader("上传色卡或颜色样板", type=['jpg', 'png', 'jpeg'])
    if ref_file:
        ref_img = Image.open(ref_file)
        st.image(ref_img, caption="参考色预览", width=200)
        t_rgb = get_ref_color_precise(ref_img)
        st.success(f"已锁定参考色: {int(t_rgb[0])}, {int(t_rgb[1])}, {int(t_rgb[2])}")

st.header("2. 待换色照片 (上传后可直接在此预览)")
uploaded_files = st.file_uploader("支持批量上传 (jpg/png)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

# 照片预览功能
if uploaded_files:
    with st.expander("🖼️ 点击预览已上传的照片", expanded=True):
        cols = st.columns(5)
        for i, f in enumerate(uploaded_files):
            with cols[i % 5]:
                st.image(f, caption=f"原图 {i+1}", use_container_width=True)

# 处理逻辑
if uploaded_files and ref_file:
    if st.button("🚀 开始 AI 精准颜色复刻"):
        t_rgb = get_ref_color_precise(Image.open(ref_file))
        res_cols = st.columns(2)
        zip_list = []
        
        for idx, file in enumerate(uploaded_files):
            try:
                res = process_perfect_match(Image.open(file), t_rgb)
                zip_list.append({"name": file.name, "img": res})
                with res_cols[idx % 2]:
                    st.image(res, caption=f"复刻结果: {file.name}", use_container_width=True)
            except Exception as e:
                st.error(f"处理第 {idx+1} 张图出错: {e}")

        if zip_list:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as f:
                for item in zip_list:
                    img_io = io.BytesIO()
                    Image.fromarray(item["img"]).save(img_io, format='JPEG', quality=95)
                    f.writestr(f"cloned_{item['name']}", img_io.getvalue())
            st.download_button("💾 下载所有结果 (ZIP)", buf.getvalue(), "output.zip")
else:
    st.info("👈 请在左侧上传参考图，然后在上方上传待处理的照片。")
