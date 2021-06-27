# Product's name: Yoga Vibe 
## An AI companion for your exciting yoga journey

## Purposes and Motivations
- Yoga is a great way to develop both physical and mental health. A few minutes doing yoga a day can relieve our minds, brings joys and happiness which will be a huge felp for us in this hard time. The need for this is even rising during this covid-19 era, even though there are online yoga classes, limitation of time is still threre.. so a self-instructed app for this is essential.
- The app provides users small practices with major postures in yoga, and each task is timed (only runs out when correct pose is dectected). Via this way, it can create a momentum, helps users get used to the routine and may be eager for more. Like a positive domino effect. And on top of it, users can use this to practice anywhere, anytime. 
- A little bit of challenges make things more exciting
- The fact that I suck at yoga and my laziness are what motivates me the most to do this project. And I'm also in hope to empower all the lazy/busy people out there to do yoga, get some of its benefits and maybe enlighten their lives a little bit
---
## Product description:
- Yoga vibe provides 2 modes for users: Learn and Practice
- Learn: Users have to hold the correct pose which is displayed on the right side for 10 seconds. If more than 3 mistakes are made, the timer will reset
- Practice: Users have to hold the correct pose which is displayed on the right side for 45 seconds. If more than 3 mistakes are made, the timer will reset
- There is a music player on the top left for user to enjoy while practicing
---
## Dataset
- https://www.kaggle.com/elysian01/yoga-pose-classification
- The dataset originally contains 2905 images of 6 major poses in Hatha Yoga. 6 poses are divided into 2 sets and each set contains 2 subsets for training and test. 
- After cleaning, there are 2783 images left for 6 classes. I've run pose estimator on the whole dataset with threshold of 0.5 => remove images with no human in it. The orignal dataset is messy and unbalance between validation and train set => Combined the two and performed data cleaning, preparation then use train_test_split with test size of 0.2.
- Pose estimator are used to extract landmark results -> 2783 arrays of shape (33,2) are saved in pickle files, labels are one-hot encoded. 
---
## Model selection
- Pose estimator: Mediapipe - Pose of Google as it is light-weighted and has good performance
- Classification: 1D CNN is applied. As the dataset is relatively small, my approach was going from a simple model then adding more complexity later and using regularizer L2, Dropout to avoid over fitting
---
## Helpful links
- Research about yoga postures classification: https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=1932&context=etd_projects 
- Mediapipe: https://google.github.io/mediapipe/solutions/pose_classification.html
---
## Further improvements:
- Data record system
- Deployment
- Posture correction
- Expanding dataset, available poses 
- Customization
---
## File structure
- My approach for the project is writing everything first in Python and later on, translate some parts to Javascripts for the Flask app
- skeleton_cnn_7.h5: Final model
- create_dataset.py: For landmarks extraction
### Python scripts and display frame using openCV:
- learning_mode.py, practice_mode.py: functions for learning and practicing modes
- live_prediction.py: Detection and classification functions
### Files to run flask app:
- app.py: flask app file
- main_prediction.py: Script with loaded model for pose classification 
- templates: HTML files 
- static: javascript, html/css and other static files
