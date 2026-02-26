import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io, zipfile

# --- 1. 颜色精准采样 ---
def get_target_lab(ref_img):
    img = np.array(ref_img.convert('RGB'))
    h, w, _ = img.shape
    # 取中心 20% 区域，避开边缘
    roi = img[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)]
    avg_rgb = np.mean(roi, axis=(0, 1)).astype(np.uint8)
    # 转为 LAB
    target_lab = cv2.cvtColor(np.uint8([[avg_rgb]]), cv2.COLOR_RGB2LAB)[0][0]
    return target_lab

# --- 2. 核心算法：补光 + 语义保护 ---
def process_core(original_img, target_lab):
    # 转为数组并记录尺寸
    rgb_img = np.array(original_img.convert('RGB'))
    h, w, _ = rgb_img.shape
    
    # A. 提取主体掩模
    mask = np.array(remove(original_img, only_mask=True))

    # B. 【建立多层防护罩】
    # 1. 皮肤防护 (YCrCb 空间最稳)
    ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    skin = cv2.dilate(skin, np.ones((5,5), np.uint8), iterations=2)

    # 2. 中性色防护 (针对白鞋、灰色器材)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    neutral = cv2.threshold(saturation, 45, 255, cv2.THRESH_BINARY_INV)[1]

    # 3. 底部防护 (保护鞋子：图片底部 15% 区域不准变色)
    bottom_shield = np.ones((h, w), dtype=np.uint8) * 255
    bottom_shield[int(h*0.85):, :] = 0 

    # 合成服装掩模
    clothes_mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin))
    clothes_mask = cv2.bitwise_and(clothes_mask, cv2.bitwise_not(neutral))
    clothes_mask = cv2.bitwise_and(clothes_mask, bottom_shield)
    clothes_mask_blur = cv2.GaussianBlur(clothes_mask, (15, 15), 0) / 255.0

    # C. 【黑色补光与色彩映射】
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # 黑色上色逻辑：如果亮度(L)太低，强制提升
    target_l = target_lab[0]
    # 对暗部进行非线性亮度提升
    l_fixed = np.where(l < 70, l * 0.4 + (target_l * 0.6), l)
    l_fixed = np.clip(l_fixed, 0, 255)

    # 颜色克隆
    a_new = np.full_like(a, target_lab[1])
    b_new = np.full_like(b, target_lab[2])

    # 合成
    merged_lab = cv2.merge([l_fixed, a_new, b_new]).astype(np.uint8)
    new_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    # D. 最终融合
    m = clothes_mask_blur[:, :, np.newaxis]
    final = (new_rgb * m + rgb_img * (1 - m)).astype(np.uint8)
    return final

# --- 3. Streamlit 界面 ---
st.set_page_config(page_title="AI 换色专业版", layout="wide")
st.title("👕 AI 服装 1:1 复刻系统 (稳健版)")

with st.sidebar:
    st.header("1. 参考颜色")
    ref_file = st.file_uploader("上传色卡/参考图", type=['jpg', 'png', 'jpeg'])
    if ref_file:
        ref_img = Image.open(ref_file)
        st.image(ref_img, caption="提取源")
        t_lab = get_target_lab(ref_img)
        st.success("颜色已锁定")

st.header("2. 待变色图片")
files = st.file_uploader("支持批量上传", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if files and ref_file:
    if st.button("🚀 开始处理"):
        t_lab = get_target_lab(Image.open(ref_file))
        cols = st.columns(2)
        results = []
        
        for idx, f in enumerate(files):
            try:
                # 核心处理，添加错误捕获防止程序崩溃
                img = Image.open(f).convert('RGB')
                res = process_core(img, t_lab)
                results.append({"name": f.name, "img": res})
                with cols[idx % 2]:
                    st.image(res, caption=f"结果: {f.name}", use_container_width=True)
            except Exception as e:
                st.error(f"跳过错误文件 {f.name}: {str(e)}")

        if results:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for r in results:
                    img_io = io.BytesIO()
                    Image.fromarray(r["img"]).save(img_io, format='JPEG', quality=95)
                    z.writestr(f"fixed_{r['name']}", img_io.getvalue())
            st.download_button("💾 下载全部结果", buf.getvalue(), "output.zip")
else:
    st.info("请先上传参考图和待变色照片。")
