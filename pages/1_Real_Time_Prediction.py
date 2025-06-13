import streamlit as st
from Home import face_rec

#to connect our app to webcam we use streamlit webrtc
#the av library is used handle video frames inside your Streamlit WebRTC application.
from streamlit_webrtc import webrtc_streamer
import av
import time 

#st.set_page_config(page_title="Real-time prediction",layout="wide")
#create header
st.subheader("Real_Time Prediction")

#retrive data from redis
with st.spinner("Retriving data from Redis DB......"):
    redis_face_db=face_rec.retrive_data(name="St_Anns_Attendance_Register")
    #display dataframe
    st.dataframe(redis_face_db)
st.success("Data successfully retrived from redis db")


#time 
#time to wait to call savelogs_redis() function 
waitTime=30 #time in sec
#set time is current time 
setTime=time.time() # Save the time when app starts
realtimepred=face_rec.RealTimePred() #call realtime prediction class 




#real time-prediiction
#streamlit-webrtc
#saving logs every 30 sec
#call back function 
def video_frame_callback(frame):
    #save logs every 30 sec 
    global setTime
    
    img = frame.to_ndarray(format="bgr24") #"bgr24"- is 3d numpy array
    
    #operation that u can perform on array
    #prediction function
    pred_img=realtimepred.prediction(img,redis_face_db,"Facial_Features",["Name","Role"],thresh=0.5)
    #to reset set time i.e. current time every 30 sec to save that data to redis db
    timenow=time.time() # Current time
    difftime=timenow-setTime
    if difftime >=waitTime:
        realtimepred.saveLogs_redis()
        setTime=time.time() #reset time
        print("Save data to Redis DB")    
    
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24") 

webrtc_streamer(key="real_time_prediction", video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)