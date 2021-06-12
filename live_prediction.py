import cv2
import os 
import mediapipe as mp 
import numpy as np 
import pickle
mp_drawing = mp.solutions.drawing_utils #all the drawing utilities
mp_pose = mp.solutions.pose #wrapping pose estimation model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import practice_mode 

#setup
data_dir = 'D:\Yoga_Companion\processed_set\data'
poses_list = os.listdir(data_dir)
poses_list.sort()
classes = {i:poses_list[i] for i in range(len(poses_list))}
#pics collection dir
collection_path = 'D:\Yoga_Companion\my_yoga_time_collection'
#load model
classifier = load_model('skeleton_cnn_4.h5')

def live_prediction(classifier, classes):
    cap = cv2.VideoCapture(0)
    #set up mediapipe
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        batch = []
        while cap.isOpened():
            ret, frame = cap.read()
            img = frame.copy()

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img.flags.writeable = False
            #detection
            results = pose.process(img)
            img.flags.writeable = True 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not results.pose_landmarks:
                    continue 
            
            #each has shape of (n, 2) with m are detected key points with x, y coordinates
            #get datasample
            lm_list = []
            #re-order joint position
            landmarks = results.pose_landmarks.landmark
            # joint_order = [landmarks[j] for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                # 11, 13, 15, 17, 19, 21, 
                # 12, 14, 16, 18, 20, 22,
                # 23, 25, 27, 29, 31, 
                # 24, 26, 28, 30, 32]]
            
            for lm in landmarks:
                lm_list.append((lm.x, lm.y))
                
        
            # batch.append(lm_list)
            # batch = np.array(batch)

            #prediction
            prediction = classifier.predict(np.array(lm_list)[np.newaxis, :, :])

            if np.amax(prediction) > 0.7:
                label = np.argmax(prediction, axis = 1)
                name = classes[int(label)]

                cv2.putText(frame, name, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 1, 255), thickness = 2)
                cv2.putText(frame, str(np.amax(prediction)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 1, 255), thickness = 2)
            
            elif np.amax(prediction) <= 0.7:
                name = 'pose undetected'
                cv2.putText(frame, name, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 1, 255), thickness = 2)
                cv2.putText(frame, str(np.amax(prediction)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 1, 255), thickness = 2)

            #draw skeleton
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color = (245, 117, 66), thickness=2,  circle_radius=2), mp_drawing.DrawingSpec(color = (245, 66, 230), thickness = 2, circle_radius=2))

            cv2.imshow('Yoga practice', frame)

            #Break when pressing ESC
            c = cv2.waitKey(1) 
            if c == 27:
                break 
            # #extra take a pic feature, will be developed more in the future if
            # elif c == 32:
            #     name = input()
            #     img_path = os.path.join(collection_path, name)
            #     cv2.imwrite(img_path,frame)
    cap.release()
    cv2.destroyAllWindows()
    return name, prediction

