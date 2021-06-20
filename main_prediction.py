import cv2
import os
import mediapipe as mp 
import numpy as np 
mp_drawing = mp.solutions.drawing_utils #all the drawing utilities
mp_pose = mp.solutions.pose #wrapping pose estimation model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from time import sleep

#setup
data_dir = 'D:\Yoga_Companion\processed_set\data'
poses_list = os.listdir(data_dir)
poses_list.sort()
classes = {i:poses_list[i] for i in range(len(poses_list))}

#load model
classifier = load_model('skeleton_cnn_4.h5')

def live_prediction(classifier, classes, image):
    global errors, counters, time, name, prediction  
    #set up mediapipe
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        frame = cv2.imread(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame.flags.writeable = False
        #detection
        results = pose.process(frame)
        frame.flags.writeable = True 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
        #each has shape of (n, 2) with m are detected key points with x, y coordinates
        #get datasample
            lm_list = []
            landmarks = results.pose_landmarks.landmark
            
            for lm in landmarks:
                lm_list.append((lm.x, lm.y))
                
            #prediction
            prediction = classifier.predict(np.array(lm_list)[np.newaxis, :, :])
            
            if np.amax(prediction) > 0.7:
                label = np.argmax(prediction, axis = 1)
                name = classes[int(label)]
            else:
                name = 'pose undetected'
        except:
            name = 'undetected'
    return name


