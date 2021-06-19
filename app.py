#pomodoro
from flask import Flask, render_template, request, jsonify
import base64
import uuid 
import re
from learning_mode import *

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/instructionsLearn')
def instructionLearn():
    return render_template('instructionsLearn.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/instructionsPractice')
def instructionPractice():
    return render_template('instructionsPractice.html')

@app.route('/practice')
def practice():
    return render_template('practice.html')

def parse_image(imgData):
    img_str = re.search(b"base64,(.*)", imgData).group(1)
    img_decode = base64.decodebytes(img_str)
    filename = "{}.jpeg".format(uuid.uuid4().hex)
    print(filename)
    with open('uploads/'+filename, "wb") as f:
        f.write(img_decode)
    # return img_decode
    return 'uploads/'+filename

# def preprocess(image):
#     image = tf.image.decode_jpeg(image, channels=3)
# #     image = tf.image.resize(image, [192, 192])
# #     # Use `convert_image_dtype` to convert to floats in the [0,1] range.
# #     image = tf.image.convert_image_dtype(image, tf.float32)
# #     image = (image*2) - 1  # normalize to [-1,1] range
# #     image = tf.image.per_image_standardization(image)
#     return image

@app.route('/send_image', methods=['POST'])
def pose_predict():
    data_image = request.get_json()
    image_path = parse_image(data_image['data-uri'].encode())
    name = live_prediction(classifier, classes, image_path)

    return jsonify({'label': name}) 
    # return jsonify({'key':'haha'})

if __name__ == "__main__":
    app.run(debug=True) 
    
# def predict(image):
#     name = live_prediction(classifier, classes, image)
#     print(name)

# image = os.path.join('D:\Yoga_Companion\my_yoga_time_collection', '76.jpeg')
# predict(image)
# print(classes)
