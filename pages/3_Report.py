import streamlit as st
from Home import face_rec
import pandas as pd

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
tab1,tab2,tab3=st.tabs(["Registered Data","Logs","Attendance Report"])

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
    
    
#adding attendance report  as tab 3
with tab3:
    st.subheader("Attendance Report")
    #load logs data as list
    logs_list=load_logs(name=name)
    
    #convert logs from list of bytes to list of strings
    convert_logs_bytes_to_string=lambda x: x.decode("utf-8")
    logs_list_string=list(map(convert_logs_bytes_to_string,logs_list))
    
    #Split the string in time log data by @ and split it to nested list
    split_string=lambda x: x.split("@")
    logs_nested_list=list(map(split_string,logs_list_string))
    
    #conv nested list to df
    logs_df=pd.DataFrame(logs_nested_list,columns=["Name","Role","Timestamp"])
    
    #IN and out time
    ##step 1 convert timestamp to date and time format and extract date 
    logs_df["Timestamp"]=pd.to_datetime(logs_df["Timestamp"])
    logs_df["Date"]=logs_df["Timestamp"].dt.date
    
    ##step 2 calculating in-time,out-time and duration 
    ###intime- is the time at which the person is first detected during that date(min timestamp of the data)
    ###outtime- is the time at which the person is last detected during that date(max timestamp of the data)
    report_df=logs_df.groupby(by=["Date","Name","Role"]).agg(
        In_time=pd.NamedAgg("Timestamp",min), #in time 
        Out_time=pd.NamedAgg("Timestamp",max) #out time
    ).reset_index()
    
    #convert in and out time to date and time format
    report_df["In_time"]=pd.to_datetime(report_df["In_time"])
    report_df["Out_time"]=pd.to_datetime(report_df["Out_time"])
    
    ###duration =intime-outtime 
    report_df["Duration"]=report_df["Out_time"]-report_df["In_time"]
    
    
    #Marking Present and absent 
    ##step 1 extract all the dates available
    all_dates=report_df["Date"].unique()
    ##step 2 extract all name and role available and remove duplicates and conv it array then to list
    name_role=report_df[["Name","Role"]].drop_duplicates().values.tolist()   
    ##step 3 create a zip file which contains unique name_role and date
    date_name_role_zip=[]
    for dt in all_dates:
        for name, role in name_role:
            date_name_role_zip.append([dt,name,role])
    
    date_name_role_zip_df=pd.DataFrame(date_name_role_zip,columns=["Date","Name","Role"])
    
    ##step 4 merge zip file with actual data i.e. left join with report_df
    date_name_role_zip_df=pd.merge(date_name_role_zip_df,report_df,how="left",on=["Date","Name","Role"])
    
    ##step 5 calculate duration in seconds and hours 
    date_name_role_zip_df["Duration_seconds"]=date_name_role_zip_df["Duration"].dt.seconds
    date_name_role_zip_df["Duration_hours"]=date_name_role_zip_df["Duration_seconds"]/(60*60)
    
    ##create a function and set range of present and absent 
    def status_marker(x):
        if pd.Series(x).isnull().all():
            return "Absent"
        elif x>=0 and x<1:
            return "Absent (less than 1 hr)"
        elif x>=1 and x<4:
            return "Half day (less than 4 hrs)"
        elif x>=4 and x<6:
            return "Half day (less than 6 hrs)"
        elif x>=6:
            return "Present"
    
    ##create status column in data_name_role_zip_df to apply sataus_marker function 
    date_name_role_zip_df["Status"]=date_name_role_zip_df["Duration_hours"].apply(status_marker)
            
        
    
    st.dataframe(date_name_role_zip_df)