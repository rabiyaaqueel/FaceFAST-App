import streamlit as st

st.set_page_config(page_title="Face_recognition_Attendence_system",layout="wide")
#create header
st.header("FaceFast_App")

#reducing the app data loading time
with st.spinner("Loading models and connecting to Redis db...."):
    import face_rec
    
st.success("Model loaded successfully")
st.success("Redis db loaded successfully")