import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai

genai.configure(api_key="AIzaSyAHdGoK3fyNHKDZ7vK1lg7gIEAbIbXIlO0")

model_gemini = genai.GenerativeModel('gemini-pro')

labels = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 'Badipala', 'Balloon_Vine', 'Bamboo',
          'Beans', 'Betel', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly',
          'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddpathre', 'Drumstick',
          'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus',
          'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass',
          'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion',
          'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose',
          'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato',
          'Tulsi', 'Turmeric', 'ashoka', 'camphor', 'kamakasturi', 'kepala']

# Load the image model
model = tf.keras.models.load_model('models/model_avg_25.h5')

# Function to get or create session state


def get_session_state():
    return st.session_state


# Initialize session state
session_state = get_session_state()

# Initialize session state variables
if 'img' not in session_state:
    session_state.img = None
if 'label' not in session_state:
    session_state.label = None


def get_herb_details(herb):
    prompt_details = f'''Assume yourself as an ayurvedic doctor and get details of {herb} in format:
    Name: {herb} \n
    Family: \n
    Description: \n
    Medicinal Properties:\n'''
    response = model_gemini.generate_content(prompt_details)
    return response.text


def get_herb_benefits(herb):
    prompt_benefits = f'''Assume yourself as an ayurvedic doctor and generate benefits of {herb} to human health and any cures to diseases in bullet points. Don't exceed 5 points.'''
    response = model_gemini.generate_content(prompt_benefits)
    return response.text


def get_disease_cures(disease):
    prompt_disease = f'''Assume yourself as an ayurvedic doctor and generate cures for {disease} using herbs from {labels} generate atleast 5 cures.If it is serious disease like cancer, thyroid, or anything that requires surgery then please sugest to consult a doctor and don't recommend cures. Don't use more than two herbs for cures instead provide disclaimer. Use only popular herbs for that particular disease.'''
    response = model_gemini.generate_content(prompt_disease)
    return response.text


def predict_image(img):
    img = tf.keras.preprocessing.image.load_img(
        img, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.sigmoid(predictions[0])
    label = labels[np.argmax(score)]
    return label


st.title("Herb Identification")

with st.sidebar:
    st.image("./plant.png")
    choice = st.radio(
        "Navigation", ["Upload Image", "Predict Herb & Description", "Herb Benefits", "Disease Diagnosis"])
    st.info("This application identifies the medicinal herb and helps cure basic diseases using herbs.")

if choice == "Upload Image":
    st.subheader("Upload Image")
    session_state.img = st.file_uploader("Choose an image...", type="jpg")
    st.markdown(
        "[Donwload the test images](https://drive.google.com/uc?export=download&id=1Hj_ybQGQYfs1qih6HwfEnd0m5Y231ceU)")
    if session_state.img is not None:
        st.image(session_state.img, caption=f"Uploaded Image",
                 width=300, use_column_width='auto')

elif choice == "Predict Herb & Description":
    st.subheader("Predict Herb & Description")
    if session_state.img is not None:
        session_state.label = predict_image(session_state.img)
        st.image(session_state.img,
                 caption=f"Predicted Image: {session_state.label}", width=300, use_column_width='auto')
        st.write(get_herb_details(session_state.label))

elif choice == "Herb Benefits":
    st.subheader("Herb Benefits")
    if session_state.img is not None and session_state.label is not None:
        st.image(session_state.img,
                 caption=f"Predicted Image: {session_state.label}", width=300, use_column_width='auto')
        st.write(get_herb_benefits(session_state.label))

elif choice == "Disease Diagnosis":
    st.subheader("Disease Diagnosis")
    predefined_options = ["Fever", "Cough", "Cold", "Head Ache", "Indigestion",
                          "Acne", "Insomnia", "Obesity", "Constipation", "Gas and Bloating", "Hair Damage"]

    option = st.selectbox(label="Select from list of diseases",
                          options=predefined_options + ["Other"])

    if option == "Other":
        custom_disease = st.text_input("Enter the disease:")
        submitted = st.button(label='Submit')
        if submitted:
            st.write(get_disease_cures(custom_disease))
    else:
        submitted = st.button(label='Submit')
        if submitted:
            st.write(get_disease_cures(option))
