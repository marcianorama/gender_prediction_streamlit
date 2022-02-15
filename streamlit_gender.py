import streamlit as st
import pickle

# parsing result
jk_label = {1:"LAKI-LAKI", 0:"PEREMPUAN"}

# load the vectorizer
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

# load the model
loaded_model = pickle.load(open('gender_multinomial_nb_model.pickle', 'rb'))

sentence = st.text_input('Input your name here:') 

if sentence:
    # make a prediction
    result = loaded_model.predict(loaded_vectorizer.transform([sentence]))
    # decode to predicted label
    result = jk_label[int(result)]
    # write to front end
    st.write("Halo,",sentence)
    st.write("Your Gender is :",result)