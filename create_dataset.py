# !pip install mediapipe opencv-python

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

#setup
data_dir = 'D:\Yoga_Companion\processed_set\data'
poses_list = os.listdir(data_dir)
poses_list.sort()
classes = {i:poses_list[i] for i in range(len(poses_list))}


#get skeleton sample from data

def get_skeleton(data_dir, classes):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        data = []
        labels = []
        for i, label in classes.items():
            path = os.path.join(data_dir, label)
            files = os.listdir(path)
            for file in files:
                img = cv2.imread(os.path.join(path, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                results = pose.process(img)
                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if not results.pose_landmarks:
                    continue 
                #get datasample
                lm_list = []
                
                #each has shape of (n, 2) with m are detected key points with x, y coordinates
                #reorder joint position
                landmarks = results.pose_landmarks.landmark
                joint_order = [landmarks[j] for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 13, 15, 17, 19, 21, 
                12, 14, 16, 18, 20, 22,
                23, 25, 27, 29, 31, 
                24, 26, 28, 30, 32]]


                for lm in joint_order:
                    lm_list.append((lm.x, lm.y))
                data.append(lm_list)
                #create 1 hot encoded label sample
                label_sample = np.zeros(6)
                label_sample[i] = 1
                labels.append(label_sample)
    return np.array(data), np.array(labels)

def save_to_pickle(data_dir, data, labels):
    data_dict = {'data':data, 'labels':labels}
    file_name = os.path.join(data_dir,'skeleton_reorder.pkl')

    open(file_name, 'wb').close()

    with open(file_name, 'wb') as f:
        pickle.dump(data_dict, f)
    print('saved')

def load_data_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f)
        data = data_dict['data']
        labels = data_dict['labels']
    return np.array(data), np.array(labels)
        

# data, labels = get_skeleton(data_dir, classes)

# save_to_pickle('D:\Yoga_Companion\skeleton_set', data, labels)

               
