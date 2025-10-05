import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import matplotlib.pyplot as plt
from PIL import Image
from deep_translator import GoogleTranslator
import random
import os

# Konfigurasi halaman
st.set_page_config(page_title="🧠 Analisis Kesehatan Mental", layout="wide", page_icon="🧠")

# --- FUNGSI & MODEL LOADING ---
@st.cache_resource
def load_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer, stop_words

model, vectorizer, stop_words = load_resources()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join([word for word in text.split() if word not in stop_words])

def terjemahkan_ke_inggris(teks_indonesia):
    try:
        return GoogleTranslator(source='id', target='en').translate(teks_indonesia)
    except Exception as e:
        st.error(f"Gagal menerjemahkan teks: {e}")
        return teks_indonesia

def predict_mental_health(text_indonesia):
    text_english = terjemahkan_ke_inggris(text_indonesia)
    st.info(f"Teks terjemahan (EN): {text_english}")
    cleaned_english = clean_text(text_english)
    if not cleaned_english: return "Teks tidak valid.", [0.5, 0.5]
    vector = vectorizer.transform([cleaned_english])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    label = "🔴 Terdeteksi Sinyal Isu Kesehatan Mental" if prediction == 1 else "🟢 Tidak Terdeteksi Sinyal Isu"
    return label, proba

# --- DATA REKOMENDASI LAGU (DENGAN NAMA FILE YANG SUDAH DIPERBAIKI) ---
REKOMENDASI_LAGU = [
    {
        "judul": "Semua Ini Untuk Apa?",
        "artis": "Donne Maula",
        "poster_file": "images/smua ini utk apa.jpeg",
        "sinopsis": "Lirik lagu ini menggambarkan perasaan lelah dan bertanya-tanya mengapa seseorang harus melalui begitu banyak kesulitan...",
        "url_youtube": "http://www.youtube.com/watch?v=7dEgPMzbpDY"
    },
    {
        "judul": "Manusia Kuat",
        "artis": "Tulus",
        "poster_file": "images/manusia kuat.jpeg",
        "sinopsis": "Lagu ini adalah anthem tentang ketahanan dan kekuatan batin...",
        "url_youtube": "http://www.youtube.com/watch?v=LHOh-R00gzI"
    },
    {
        "judul": "Diri",
        "artis": "Tulus",
        "poster_file": "images/diri.jpeg",
        "sinopsis": "Lagu ini mengajak pendengar untuk berdamai dengan diri sendiri, memaafkan kesalahan masa lalu, dan mencintai diri sendiri...",
        "url_youtube": "http://www.youtube.com/watch?v=fsGcUWiylW8"
    },
    {
        "judul": "Satu-Satu",
        "artis": "Idgitaf",
        "poster_file": "images/satu satu.jpeg",
        "sinopsis": "Lagu ini bercerita tentang proses penyembuhan setelah mengalami kehancuran...",
        "url_youtube": "http://www.youtube.com/watch?v=jlfMHjylvGA"
    },
    {
        "judul": "Mengudara",
        "artis": "Idgitaf",
        "poster_file": "images/mengudara.jpeg",
        "sinopsis": "Sebuah lagu tentang dukungan dan doa untuk seseorang yang 'mengudara jauh'...",
        "url_youtube": "http://www.youtube.com/watch?v=G5QqADbG7Vs"
    },
    {
        "judul": "Sudah",
        "artis": "Ardhito Pramono",
        "poster_file": "images/sudah.jpeg",
        "sinopsis": "Sebuah lagu yang sering diinterpretasikan sebagai refleksi tentang penerimaan dan melanjutkan hidup.",
        "url_youtube": "http://www.youtube.com/watch?v=nkJnteauOAY"
    },
    {
        "judul": "Daur Hidup",
        "artis": "Donne Maula",
        "poster_file": "images/daur hidup.jpeg",
        "sinopsis": "Lagu ini merefleksikan siklus kehidupan, di mana jiwa terus bertahan melalui berbagai cerita...",
        "url_youtube": "http://www.youtube.com/watch?v=HEWBFp4raoQ"
    },
    {
        "judul": "Sekuat Sesakit",
        "artis": "Idgitaf",
        "poster_file": "images/sekuat sesakit.jpeg",
        "sinopsis": "Lagu ini menggambarkan seseorang yang selalu terlihat tersenyum meski memikul beban berat...",
        "url_youtube": "http://www.youtube.com/watch?v=idCr3bbDm0g"
    },
    {
        "judul": "Tenang",
        "artis": "Yura Yunita",
        "poster_file": "images/tenang.jpeg",
        "sinopsis": "Lagu ini mengungkapkan kegelisahan dan kerinduan akan ketenangan yang tak kunjung datang...",
        "url_youtube": "http://www.youtube.com/watch?v=hoZEi4zina4"
    },
    {
        "judul": "Rehat",
        "artis": "Kunto Aji",
        "poster_file": "images/rehat.jpeg",
        "sinopsis": "Pengingat lembut untuk beristirahat, menenangkan diri, dan memvalidasi perasaanmu. Tidak apa-apa untuk tidak baik-baik saja.",
        "url_youtube": "http://www.youtube.com/watch?v=yNcGtKAacts"
    },
    {
        "judul": "Menjadi Manusia",
        "artis": "Donne Maula",
        "poster_file": "images/menjadi manusia.jpeg",
        "sinopsis": "Sebuah introspeksi tentang nilai diri dan perjuangan untuk kembali menjadi manusia seutuhnya...",
        "url_youtube": "http://www.youtube.com/watch?v=LGk27ul-bi8"
    }
]


# --- MANAJEMEN NAVIGASI & STATE ---
if "page" not in st.session_state: st.session_state.page = "🏠 Landing Page"
if "result" not in st.session_state: st.session_state.result = None
if "show_music_player" not in st.session_state: st.session_state.show_music_player = False

def set_page(page_name):
    st.session_state.page = page_name
    # Reset state saat pindah halaman utama
    if page_name == "🔍 Prediksi":
        st.session_state.result = None
        st.session_state.show_music_player = False

# --- UI ---
with st.sidebar:
    st.title("🧭 Navigasi")
    if st.button("🏠 Halaman Utama", use_container_width=True):
        set_page("🏠 Landing Page")
    if st.button("🔍 Mulai Analisis", use_container_width=True):
        set_page("🔍 Prediksi")
    st.markdown("---")
    st.markdown("Dibuat dengan ❤️ olehmu")

if st.session_state.page == "🏠 Landing Page":
    st.image("header.jpg", use_column_width=True)
    st.markdown("<h1 style='text-align: center;'>Selamat Datang di Analisis Sinyal Kesehatan Mental</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Tuliskan perasaanmu dan biarkan AI membantu menganalisisnya.</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.button("🚀 Lanjut ke Analisis", on_click=set_page, args=("🔍 Prediksi",), use_container_width=True)

elif st.session_state.page == "🔍 Prediksi":
    st.markdown("<h1 style='text-align:center;'>🧠 Analisis Isu Kesehatan Mental</h1>", unsafe_allow_html=True)
    user_input = st.text_area("Ceritakan perasaanmu:", height=150)

    if st.button("🔍 Analisis Sekarang", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Mohon isi dulu kolom perasaanmu.")
        else:
            with st.spinner('Menganalisis teks...'):
                result, prob = predict_mental_health(user_input)
                st.session_state.result = result
                st.session_state.show_music_player = False # Reset player saat analisis baru

    if st.session_state.result:
        st.success(f"**Hasil Analisis:** {st.session_state.result}")
        if st.session_state.result.startswith("🔴"):
            st.markdown("---")
            st.markdown("#### 💡 Sesi Intervensi")
            st.info("Aku turut merasakan beratnya harimu. AI Composer saya telah menciptakan sebuah melodi untukmu.")
            
            playlist_ai = [f"audio/{f}" for f in os.listdir("audio") if f.endswith('.mp3')]
            if playlist_ai:
                lagu_ciptaan_ai = random.choice(playlist_ai)
                st.audio(lagu_ciptaan_ai, format='audio/mp3', start_time=0)
                st.markdown("> *Pejamkan matamu, tarik napas dalam-dalam, dan biarkan alunan musik ini menemanimu sejenak...*")
            else:
                st.warning("Tidak ada file musik AI di folder 'audio'.")
            
            st.markdown("---")
            st.button("Lihat Rekomendasi Lagu Tambahan 🎵", on_click=set_page, args=("🎵 Rekomendasi",), use_container_width=True)
        else:
            st.balloons()
            st.markdown("---")
            st.markdown("#### ✨ Terima Kasih Sudah Berbagi")

elif st.session_state.page == "🎵 Rekomendasi":
    st.markdown("<h1 style='text-align:center;'>Rekomendasi Lagu untukmu</h1>", unsafe_allow_html=True)
    st.markdown("<p>Berikut adalah beberapa lagu pilihan yang lirik dan melodinya dapat membantu merefleksikan dan menguatkan diri.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Loop untuk menampilkan semua rekomendasi
    for i, lagu in enumerate(REKOMENDASI_LAGU):
        st.markdown(f"### {i+1}. {lagu['judul']} - {lagu['artis']}")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if os.path.exists(lagu['poster_file']):
                st.image(lagu['poster_file'], use_container_width=True)
        with col2:
            st.markdown(f"> *{lagu['sinopsis']}*")
            st.link_button("Dengarkan di YouTube ⏯️", lagu['url_youtube'])
        st.markdown("---") # Garis pemisah antar lagu

    st.button("Kembali ke Halaman Analisis", on_click=set_page, args=("🔍 Prediksi",), use_container_width=True)