import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras import preprocessing
import time


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Skin Diseases Detection')

st.markdown("Prediction : Type of Diseases")

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                # plt.imshow(image)
                # plt.axis("off")

                predictions = predict(image)

                time.sleep(1)
                st.success('Classified')
                st.write(predictions, unsafe_allow_html=True)


## this code for format tflite file
def predict(image):
    model_path = "lite_model.tflite" 

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    
    # Preprocess the image to match the input shape of the model
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image, dtype=np.float32) 
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = np.array(output_data[0])
    print(probabilities)

    # Define the labels for the 8 skin diseases
    labels = {
        0: "VI-chickenpox <br> The remedies are :<br> For chickenpox, consider antiviral medications for severe cases. Over-the-counter pain relievers like acetaminophen can manage fever, while calamine lotion soothes itching. Cool baths or wet compresses can provide further relief from discomfort.",
        1: "VI-shingles <br> The remedies are :<br> Remedies for VI-shingles typically involve antiviral medications to reduce the severity and duration of symptoms. Pain relievers like acetaminophen or ibuprofen can help manage discomfort, while applying cool compresses may soothe itching and promote healing.",
        2: "BA-cellulitis <br> The remedies are :<br> Treatment for BA-cellulitis often includes oral antibiotics to fight the bacterial infection. Elevating the affected area and applying warm compresses can help reduce pain and swelling. It's crucial to complete the full course of antibiotics as prescribed by a healthcare professional.",
        3: "BA-impetigo <br> The remedies are :<br> For BA-impetigo, treatments typically involve topical or oral antibiotics to eliminate the bacterial infection. Keeping the affected area clean and applying warm compresses can also help promote healing and reduce discomfort.",
        4: "FU-athlete-foot <br> The remedies are :<br> For FU-athlete's foot, over-the-counter antifungal creams or sprays are often effective. Keeping the affected area clean and dry, wearing breathable socks and shoes, and avoiding walking barefoot in public areas can help prevent recurrence.",
        5: "FU-nail-fungus <br> The remedies are :<br> For FU-nail fungus, topical antifungal treatments or oral medications may be prescribed by a healthcare professional. It's essential to keep the affected nails trimmed and dry, and to avoid sharing nail clippers or shoes to prevent spreading the infection.",
        6: "FU-ringworm <br> The remedies are :<br> For FU-ringworm, antifungal creams or ointments are commonly used to treat the infection. Keeping the affected area clean and dry, and avoiding sharing personal items such as towels or clothing, can help prevent its spread.",
        7: "PA-cutaneous-larva-migrans <br> The remedies are :<br> For PA-cutaneous larva migrans, topical antiparasitic medications like thiabendazole or albendazole are typically prescribed. Keeping the affected area clean and covered, and avoiding walking barefoot in areas where the parasite may be present, can help prevent further infection."
    }

    label_to_probabilities = []

    for i, probability in enumerate(probabilities):
        label_to_probabilities.append([labels[i], float(probability)])

    
    label_to_probabilities = sorted(label_to_probabilities, key=lambda element: element[1], reverse=True)

    
    result = f"{label_to_probabilities[0][0]}"
    
    return result



if __name__ == "__main__":
    main()