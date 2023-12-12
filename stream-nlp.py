import pickle
import streamlit as st
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import pytesseract
from PIL import Image
from googletrans import Translator  # Import Translator from googletrans library

# LOAD SAVE MODEL
model_fraud = pickle.load(open('model_fraud.sav', 'rb'))

# Load TF-IDF vectorizer with vocabulary
loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))

# Set page title and description
st.title('Prediksi SMS Penipuan')
st.write('Aplikasi ini digunakan untuk mendeteksi apakah SMS adalah penipuan atau tidak.')

# Splash Screen
with st.spinner("Sedang memuat..."):
    time.sleep(2)

# Create a sidebar for additional options or information
st.sidebar.header('Tentang Aplikasi')
st.sidebar.write('Aplikasi ini menggunakan model Text Mining untuk mendeteksi penipuan dalam SMS.')

# Create a text input field for user input
clean_text = st.text_area('Masukkan Teks SMS', '')

# Create a file uploader for images
uploaded_image = st.file_uploader("Upload Gambar SMS", type=["jpg", "png", "jpeg"])

# Create buttons for prediction and translation
if st.button('Deteksi Penipuan'):
    if clean_text:
        loaded_vec.fit([clean_text])
        predict_fraud = model_fraud.predict(loaded_vec.transform([clean_text]))[0]

        if predict_fraud == 0:
            fraud_detection = 'SMS Normal'
        elif predict_fraud == 1:
            fraud_detection = 'SMS Penipuan'
        else:
            fraud_detection = 'SMS Promo'

        st.subheader('Hasil Deteksi Teks:')
        st.write(fraud_detection)
    else:
        st.warning('Masukkan teks SMS terlebih dahulu.')

if st.button('Terjemahkan Gambar'):
    if uploaded_image:
        # Use Tesseract to extract text from the uploaded image
        image = Image.open(uploaded_image)
        text_from_image = pytesseract.image_to_string(image)

        st.subheader('Hasil Deteksi Teks dari Gambar SMS:')
        st.write(text_from_image)

        # Translate the text to English
        translator = Translator()
        translated_text = translator.translate(text_from_image, src='auto', dest='en')

        #st.subheader('Hasil Terjemahan:')
        #st.write(translated_text.text)  # Display the translated text

        # Deteksi penipuan pada teks yang diterjemahkan
        loaded_vec.fit([translated_text.text])
        predict_fraud = model_fraud.predict(loaded_vec.transform([translated_text.text]))[0]

        if predict_fraud == 0:
            fraud_detection = 'SMS Normal'
        elif predict_fraud == 1:
            fraud_detection = 'SMS Penipuan'
        else:
            fraud_detection = 'SMS Promo'

       

        # Menerjemahkan teks ke bahasa Indonesia jika asal bahasa Inggris
        if translated_text.src == 'en':
            translator_id = Translator()
            translated_to_indonesian = translator_id.translate(translated_text.text, src='en', dest='id')

            st.subheader('Terjemahan ke Bahasa Indonesia:')
            st.write(translated_to_indonesian.text)

        # Menerjemahkan teks ke bahasa Inggris jika asal bahasa Indonesia
        elif translated_text.src == 'id':
            translator_en = Translator()
            translated_to_english = translator_en.translate(translated_text.text, src='id', dest='en')

            st.subheader('Terjemahan ke Bahasa Inggris:')
            st.write(translated_to_english.text)

    st.subheader('Hasil Deteksi Penipuan:')
    st.write(fraud_detection)
