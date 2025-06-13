#import libraries
import pandas as pd
import numpy as np
import cv2
import os


import redis
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise #pairwise module contains functions used to compute pairwise distances and similarities between samples.


#to calculate at what time a face is detected
import time #Getting the current time in seconds: time.time
from datetime import datetime #datetime is a class inside the module that gives the current date and time and allows formatting, comparison, etc.

#connect to redis client i.e database
#connecting python to redis database
r=redis.StrictRedis(host="redis-11498.c90.us-east-1-3.ec2.redns.redis-cloud.com",port=11498,password="YUMGPm17tELn3590Gf5K53CAYmKn6tpC")



#configure insight face model  or loading the model
#configure the providers
# we are using CPU and we are using baffalo_sc model 
faceapp=FaceAnalysis(name="buffalo_sc",
                     root="./Insightface_models/models",
                     provider="CPUExecutionProvide"
)
#prepare face app
faceapp.prepare(ctx_id=0,det_size=(640,640),det_thresh=0.5)
#warning- dont keep det_thresh < 0.3


   
#step 4 retrive data from redis 
def retrive_data(name):
    #retrive data from redis 
    retrive_dic=r.hgetall(name)
    retrive_series=pd.Series(retrive_dic) #Converts the dictionary to a Pandas Series
    #Converts each byte into a NumPy array of float32 using np.frombuffer
    retrive_series=retrive_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index=retrive_series.index #Updates the index of the series from bytes to proper strings
    index=list(map(lambda x: x.decode(),index)) #Decodes each byte index to a human-readable string
    retrive_series.index=index #Updates the index of the series from bytes to proper strings
    retrive_df=retrive_series.to_frame().reset_index() #Converts the Series to a DataFrame and resets the index
    retrive_df.columns=["Name_Role","Facial_Features"]
    #split name and role
    retrive_df[["Name","Role"]]=retrive_df["Name_Role"].apply(lambda x:x.split("@")).apply(pd.Series)
    return retrive_df[["Name","Role","Facial_Features"]]




#ML search algorithm:cosine similarity
#to find person name using cosine similarity
#to find person name using cosine similarity
def ml_search_algorithm(dataframe,feature_column,test_vector,name_role=["Name","Role"],thresh=0.5):
    """
    cosine similarity 
    """
    #step 1 take the dataframe
    dataframe=dataframe.copy()
    #step 2 convert facial_features from extracted data i.e dataframe to array
    x_list=dataframe[feature_column].tolist()
    x=np.asarray(x_list)
        
    #step 3 convert the test embeddings to row vector or column vector 
    #y=test_vector.reshape(1,-1)
    
    #step 4 cal cosine similarity using pairwise function 
    similar=pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr=np.array(similar).flatten()
    dataframe["cosine"]=similar_arr
    
    #step 5 -taking values greater than 0.5
    data_filter=dataframe.query(f"cosine >= {thresh}")


    #step 6 -finding max value from values that are greater than 0.5 and get perosn name

    if len(data_filter) > 0:
        data_filter.reset_index(drop=True,inplace=True)
        argmax=data_filter["cosine"].argmax()
        person_name, person_role=data_filter.loc[argmax][name_role]
    else:
        person_name="Unknown"
        person_role="Unknown"
        
    
    return person_name, person_role

#real time prediction 
##saving the time logs every minute
class RealTimePred:
    def __init__(self):
        self.logs=dict(name=[],role=[],current_time=[])
    def reset_dict(self):
        self.logs=dict(name=[],role=[],current_time=[])
    
    #save logs data in redis db
    def saveLogs_redis(self):
        #step 1 create a logs dataframe 
        dataframe=pd.DataFrame(self.logs)
        #step 2 drop the duplicate information- gonna take only (distinct name)
        ##if a person name is detected 10 times in a min we gonna store that person name only ones in a min known as (distinct name)
        dataframe.drop_duplicates("name",inplace=True)
        #step 3 push data to redis db 
        #encode the data by conv the name,role,current_time to list then concatenate the name,role and current_time (as we save one variable in list)
        name_list=dataframe["name"].tolist()
        role_list=dataframe["role"].tolist()
        ctime_list=dataframe["current_time"].tolist()
        encoded_data=[]
        for name,role,ctime in zip(name_list,role_list,ctime_list):
            if name != "Unknown":
                concat_string=f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)
        
        #step 4 then we have to check that we have all the predictions in that particular minute and push it to redis db
        if len(encoded_data)>0:
            r.lpush("attendance:logs",*encoded_data)
        
        #step 5 reset the dict
        self.reset_dict()
    #predicting name,date and time of the person 
    def prediction(self,img_test,dataframe,feature_column,name_role=["Name","Role"],thresh=0.5):
            #getting time
            current_time=str(datetime.now())
            
            ##feature extraction
            res_test=faceapp.get(img_test)
            test_copy=img_test.copy()
            for res in res_test:
                x1,y1,x2,y2=res["bbox"].astype(int)
                embeddings=res["embedding"]
                #pass embeddings as test_vector in ml search algorithm
                person_name,person_role=ml_search_algorithm(dataframe,feature_column,test_vector=embeddings,name_role=name_role,thresh=thresh)
            
                if person_name=="Unknown":
                    color=(0,0,255)
                else:
                    color=(0,255,0)
                cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
                text_gen=person_name
                cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
                cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
                
                #save info in logs dict
                self.logs["name"].append(person_name)
                self.logs["role"].append(person_role)
                self.logs["current_time"].append(current_time)
            return test_copy
        
        
###Registration Form
#apply image to insightface model sc and get info about the image
class RegistrationForm:
    def __init__(self):
        self.sample=0
    def reset(self):
        self.sample=0
    def get_embeddings(self,frame):
        result=faceapp.get(frame,max_num=1) #as i want to detect one face
        embeddings=None
        for res in result:
            self.sample+=1
            x1,y1,x2,y2=res["bbox"].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            
            #put how many samples collected in video captured
            text=f"samples={self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,255,0),2)

            #facial features
            embeddings=res["embedding"]
        return frame,embeddings
    def save_data_in_redis_db(self,name,role):
        #Validation name
        if name is not None and name.strip() != "":
            key = f"{name}@{role}"
        else:
            return "name_false"
        
        #if face_embedding.txt not exist
        if "face_embedding.txt" not in os.listdir():
            return "file_false"
        
        #step 1 load face_embedding.txt file 
        x_array= np.loadtxt("face_embedding.txt",dtype=np.float32)
        
        #step 2 conv face_emb into array(proper shape)
        ##no of samples we have received 
        received_samples=int(x_array.size/512) #512 each face have 512 face_embeddings
        x_array=x_array.reshape(received_samples,512) #reshape each sample to 512 embeddings
        x_array=np.asarray(x_array)#conv face emb into array
        
        #step 3 take mean of face_embeddings and then conv mean to bytes
        x_mean=x_array.mean(axis=0)
        x_mean=x_mean.astype(np.float32)
        x_mean_bytes=x_mean.tobytes()
        
        #step 4 load to redis db
        ##saving in redis hashs it stire data in key:value and in bytes
        r.hset(name="St_Anns_Attendance_Register",key=key,value=x_mean_bytes)
        
        #
        os.remove("face_embedding.txt")
        self.reset()
        return True      