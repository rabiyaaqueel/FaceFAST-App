import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from aiortc import RTCConfiguration, RTCIceServer

#to connect our app to webcam we use streamlit webrtc
#the av library is used handle video frames inside your Streamlit WebRTC application.
from streamlit_webrtc import webrtc_streamer
import av

#st.set_page_config(page_title="Registration Form",layout="wide")
#create header
st.subheader("Registration Form")

#initializing registration form
registration_form=face_rec.RegistrationForm()

#step 1 collect person name and role 
##we will create a form
person_name=st.text_input(label="Name",placeholder="First and Last Name")
role=st.selectbox(label="Select your Role",options=("Student","Teacher"))




#step 2 collect face sample i.e. collect facial embeddings of that perosn 
def video_callback_func(frame):
    img = frame.to_ndarray(format="bgr24") #"bgr24"- is 3d numpy array
    rec_img,embedding=registration_form.get_embeddings(img)
    
    #2 step process to save embeddings to redis db
    #step 1 save embeddings to local computer as txt 
    if embedding is not None:
        with open("face_embedding.txt",mode="ab") as f: #opening a txt doc as face_embedding and save in ab-append the value in bytes
            np.savetxt(f,embedding) #save txt file as name-f and save embedding data
    
    return av.VideoFrame.from_ndarray(rec_img, format="bgr24") 


# ✅ Add RTC Configuration properly here
rtc_config = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

webrtc_streamer(key="registration", video_frame_callback=video_callback_func,rtc_configuration=rtc_config)


#step 3 store facial embeddings data in Redis db


if st.button("Submit"):
    return_val=registration_form.save_data_in_redis_db(person_name,role)
    if return_val==True:
        st.success(f"{person_name} Resgistered successfully")
    elif return_val=="name_false":
        st.error("Please enter the name:Name cannot be empty or spaces")
        
    elif return_val=="file_false":
        st.error("face_embedding.txt is not found. please refresh the page and execute again")