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
import timeit
import threading

#setup
data_dir = 'D:\Yoga_Companion\processed_set\data'
poses_list = os.listdir(data_dir)
poses_list.sort()
classes = {i:poses_list[i] for i in range(len(poses_list))}

#load model
classifier = load_model('skeleton_cnn_4.h5')

#practice mode
poses_lesson = ['mountain', 'downdog', 'warrior1', 'warrior2', 'goddess', 'tree']
poses_lesson = ['mountain','tree']
counters = 0
errors = 0
pose_index = 0 
time = 10
status = False
def evaluate_pose(name, frame, cap):
    global counters, errors, time, pose_index, status

    if name == poses_lesson[pose_index]:
        counters += 1 
        if counters == 10:
            cv2.putText(frame, 'finished', (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 1, 255), thickness = 2)
            switch_pose(frame, cap)
            # sleep(1)
        elif counters < 10: 
            min, sec = divmod(time, 60)
            timer = '{:02d}:{:02d}'.format(min, sec)
            time -= 1
            sleep(1)
            cv2.putText(frame, timer, (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 1, 255), thickness = 2)
    else:
        errors += 1 
        if errors == 5:
            time = 10 
            errors = 0 
            counters = 0
            return live_prediction(classifier, classes)
    
def switch_pose(frame, cap): #thread = False until evaluate done?
    global errors, counters, time, pose_index, status
    if pose_index == 1:
        status = True
        # cap.release()
        # cv2.destroyAllWindows()
    elif pose_index <1:
        errors = 0
        counters = 0 
        pose_index += 1 
        cv2.putText(frame, poses_lesson[pose_index], (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 1, 255), thickness = 2)
        time = 10

def live_prediction(classifier, classes):
    global errors, counters, time 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #set up mediapipe
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        while cap.isOpened():
            # global frame
            ret, frame = cap.read()
            # img = frame.copy()

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame.flags.writeable = False
            #detection
            results = pose.process(frame)
            frame.flags.writeable = True 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not results.pose_landmarks:
                    continue 
            
            #each has shape of (n, 2) with m are detected key points with x, y coordinates
            #get datasample
            lm_list = []
            landmarks = results.pose_landmarks.landmark
            
            for lm in landmarks:
                lm_list.append((lm.x, lm.y))
                
            global name, prediction
            #prediction
            prediction = classifier.predict(np.array(lm_list)[np.newaxis, :, :])
            
            if np.amax(prediction) > 0.7:
                label = np.argmax(prediction, axis = 1)
                name = classes[int(label)]
                cv2.putText(frame, name, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 1, 255), thickness = 2)
                cv2.putText(frame, str(np.amax(prediction)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 1, 255), thickness = 2)
            elif np.amax(prediction) <= 0.7:
                name = 'pose undetected'
            
            #draw skeleton
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color = (245, 117, 66), thickness=2,  circle_radius=2), mp_drawing.DrawingSpec(color = (245, 66, 230), thickness = 2, circle_radius=2))
            
            #get result and evaluate, start timer
            evaluate_pose(name, frame, cap)

            cv2.imshow('Yoga practice', frame)
            c = cv2.waitKey(1) 
            if c == 27:
                break 
            if status:
                break

    cap.release()
    cv2.destroyAllWindows()

live_prediction(classifier, classes)



