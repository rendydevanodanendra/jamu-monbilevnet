import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Rimpang Jamu", page_icon="🌿")

st.title("🌿 Klasifikasi Irisan Rimpang Jamu Madura")
st.write("Unggah gambar irisan rimpang (Jahe, Kencur, Kunyit, Lengkuas, atau Temulawak) dan sistem akan menebak jenisnya!")

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_rimpang.h5')
    return model

with st.spinner('Memuat Model...'):
    model = load_model()

# Label kelas sesuai urutan alfabetis folder Anda
CLASS_NAMES = ['Jahe', 'Kencur', 'Kunyit', 'Lengkuas', 'Temulawak']

# Upload File
uploaded_file = st.file_uploader("Pilih gambar rimpang...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan Gambar
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    st.write("")
    st.write("Memproses...")
    
    # Preprocessing Gambar
    # Konversi ke RGB jika formatnya RGBA (PNG)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0 # Normalisasi
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
    
    # Prediksi
    # ... (kode di atasnya tetap sama)
    
    # Prediksi
    predictions = model.predict(img_array)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = predictions[predicted_class_index] * 100
    
    # Tampilkan Hasil Utama
    st.success(f"### Prediksi: **{predicted_class}**")
    st.info(f"Tingkat Kepercayaan: **{confidence:.2f}%**")
    
    # Tampilkan Grafik Probabilitas Semua Kelas
    st.write("---")
    st.write("**Detail Probabilitas:**")
    
    # Membuat visualisasi bar chart menggunakan Streamlit bawaan
    import pandas as pd
    
    chart_data = pd.DataFrame(
        {"Probabilitas (%)": predictions * 100},
        index=CLASS_NAMES
    )
    
    st.bar_chart(chart_data)
