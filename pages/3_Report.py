import streamlit as st
from Home import face_rec

#st.set_page_config(page_title="Reporting",layout="wide")
#create header
st.subheader("Reports")

#retrive logs data and show that in reports
#extract data from redis list
name="attendance:logs"
def load_logs(name,end=-1):
    logs_list=face_rec.r.lrange(name,start=0,end=end)
    return logs_list

#creating tabs to show info of logs and data in diff tabs
tab1,tab2=st.tabs(["Registered Data","Logs"])

#adding refresh button to refresh data in reports in tab 1
with tab1:
    if st.button("Refresh Data"):
        #retrive data from redis
        with st.spinner("Retriving data from Redis DB......"):
            redis_face_db=face_rec.retrive_data(name="St_Anns_Attendance_Register")
            #display dataframe
            st.dataframe(redis_face_db[["Name",'Role']])
    

#adding refresh button to refresh logs in tab 2
with tab2:
    if st.button("Refresh Logs"):
        st.write(load_logs(name=name))
    
    
