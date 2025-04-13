import cv2
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
from PIL import Image

# -------------------------------
# Shade suggestions based on undertone
shade_suggestions = {
    "Warm": {
        "Lipstick Shades": ["Coral", "Terracotta", "Warm Red", "Peach"],
        "Blush Shades": ["Peach", "Apricot"]
    },
    "Cool": {
        "Lipstick Shades": ["Berry", "Rose", "Mauve", "Cool Red"],
        "Blush Shades": ["Pink", "Rosy"]
    },
    "Neutral": {
        "Lipstick Shades": ["Nude", "Rosewood", "Soft Pink"],
        "Blush Shades": ["Soft Rose", "Natural Pink"]
    }
}

# -------------------------------
def detect_skin(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin = cv2.bitwise_and(image, image, mask=mask)
    return skin

def get_dominant_colors(image, k=3):
    img = cv2.resize(image, (100, 100))
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def classify_undertone(colors):
    avg_color = np.mean(colors, axis=0)
    r, g, b = avg_color

    if r > g and r > b:
        return "Warm"
    elif b > r and b > g:
        return "Cool"
    else:
        return "Neutral"

# -------------------------------
# Streamlit App Interface
st.set_page_config(page_title="Skin Tone Analyzer | MARS AI", layout="centered")
st.title("🌟 AI Skin Tone & Shade Recommender")
st.markdown("Upload a selfie to detect your **skin undertone** and get **lipstick/blush shade suggestions**!")

uploaded_file = st.file_uploader("📤 Upload your photo (front face, good lighting)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    skin = detect_skin(img_bgr)
    dominant_colors = get_dominant_colors(skin, k=3)
    undertone = classify_undertone(dominant_colors)

    st.success(f"🎯 Detected Skin Undertone: **{undertone}**")

    st.subheader("💋 Recommended Shades:")
    for category, shades in shade_suggestions[undertone].items():
        st.markdown(f"**{category}:** {', '.join(shades)}")

    st.subheader("🎨 Detected Skin Tones:")
    cols = st.columns(len(dominant_colors))
    for i, col in enumerate(cols):
        color_rgb = dominant_colors[i][::-1].astype(int)
        hex_color = '#%02x%02x%02x' % tuple(color_rgb)
        col.markdown(f"<div style='background-color:{hex_color};height:60px;border-radius:8px'></div>", unsafe_allow_html=True)
        col.markdown(f"RGB: {tuple(color_rgb)}")

    st.subheader("🖼️ Processed Image:")
    st.image(skin, channels="BGR", caption="Detected Skin Region")

else:
    st.info("Please upload an image to get started.")

