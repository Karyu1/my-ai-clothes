import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io, zipfile

# --- 1. 颜色提取与校准 ---
def get_target_features(ref_img):
    img = np.array(ref_img.convert('RGB'))
    h, w, _ = img.shape
    roi = img[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    target_rgb = np.mean(roi, axis=(0, 1))
    # 转为 LAB 空间以获得精准亮度与色度
    target_lab = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2LAB)[0][0]
    return target_lab

# --- 2. 核心算法：语义保护 + 黑色补光 ---
def process_pro_v4(original_img, target_lab):
    rgb_img = np.array(original_img.convert('RGB'))
    h, w, _ = rgb_img.shape
    
    # A. 提取主体掩模
    with st.spinner("AI 正在分析衣物边界..."):
        full_mask = np.array(remove(original_img, only_mask=True))

    # B. 【多重智能保护层】
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
    
    # 1. 精准肤色保护 (脸部、手部)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin_mask = cv2.dilate(skin_mask, np.ones((7, 7), np.uint8), iterations=2)

    # 2. 中性色锁定 (重点解决鞋子和医疗器材染色)
    # 白色和灰色物体饱和度极低，识别并排除
    s_channel = hsv[:, :, 1]
    neutral_mask = cv2.threshold(s_channel, 40, 255, cv2.THRESH_BINARY_INV)[1]

    # 3. 底部地理位置保护 (针对鞋子)
    geo_protect = np.ones((h, w), dtype=np.uint8) * 255
    geo_protect[int(h*0.85):, :] = 0 

    # C. 合成服装纯净掩模
    clothes_mask = cv2.bitwise_and(full_mask, cv2.bitwise_not(skin_mask))
    clothes_mask = cv2.bitwise_and(clothes_mask, cv2.bitwise_not(neutral_mask))
    clothes_mask = cv2.bitwise_and(clothes_mask, geo_protect)
    
    # 边缘平滑处理，防止杂色边框
    clothes_mask = cv2.GaussianBlur(clothes_mask, (15, 15), 0) / 255.0

    # D. 【Lab 空间亮度重构：解决黑色不上色】
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab_img)

    # 核心算法：针对黑衣提升亮度 L，使其能展示 a/b 颜色信息
    l_target = target_lab[0]
    # 如果原图亮度低，则向目标亮度大幅靠拢
    l_new = np.where(l < 80, l * 0.3 + (l_target * 0.7), l)
    l_new = np.clip(l_new, 0, 255)

    # 强制克隆色调 A/B
    a_new = np.full_like(a, target_lab[1])
    b_new = np.full_like(b, target_lab[2])

    # 重组色彩
    merged_lab = cv2.merge([l_new, a_new, b_new]).astype(np.uint8)
    new_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    # E. 最终融合
    m = clothes_mask[:, :, np.newaxis]
    final = (new_rgb * m + rgb_img * (1 - m)).astype(np.uint8)
    return final

# --- 3. Streamlit 界面设计 ---
st.set_page_config(page_title="终极服装换色器", layout="wide")
st.title("👗 AI 服装颜色 1:1 精准复刻 (v4 终极版)")
st.write("已针对 **黑色衣服不上色** 和 **鞋子/道具误伤** 进行了底层优化。")

with st.sidebar:
    st.header("🎨 参考色设定")
    ref_file = st.file_uploader("上传目标颜色图", type=['jpg', 'png', 'jpeg'])
    if ref_file:
        ref_img = Image.open(ref_file)
        st.image(ref_img, caption="色卡/参考图")
        target_lab = get_target_features(ref_img)
        st.success("目标颜色特征已锁定")

st.header("📸 待换色照片管理")
uploaded_files = st.file_uploader("支持批量上传 (jpg/png)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.subheader("1. 检查上传原图")
    preview_cols = st.columns(6)
    for i, f in enumerate(uploaded_files):
        preview_cols[i % 6].image(f, use_container_width=True)

if uploaded_files and ref_file:
    if st.button("🚀 开始 AI 精准换色"):
        results = []
        res_cols = st.columns(2)
        
        for idx, file in enumerate(uploaded_files):
            try:
                res_img = process_pro_v4(Image.open(file), target_lab)
                results.append({"name": file.name, "img": res_img})
                with res_cols[idx % 2]:
                    st.image(res_img, caption=f"完成: {file.name}", use_container_width=True)
            except Exception as e:
                st.error(f"处理 {file.name} 时遇到错误: {str(e)}")

        if results:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as f:
                for r in results:
                    img_io = io.BytesIO()
                    Image.fromarray(r["img"]).save(img_io, format='JPEG', quality=95)
                    f.writestr(f"fixed_{r['name']}", img_io.getvalue())
            st.download_button("💾 下载全部 (ZIP 压缩包)", zip_buf.getvalue(), "results.zip")
else:
    st.info("请先上传参考颜色，再上传需要变色的照片。")
