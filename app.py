import numpy as np
from flask import Flask, request, render_template
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.python.keras.utils import array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.layers import Activation, Dropout, Flatten, Dense
app = Flask(__name__)

dic = {0: 'Benign Nevi', 1: 'Melanoma'}

base_model8 = tf.keras.applications.InceptionResNetV2(weights=None, include_top=False)

# add a global spatial average pooling layer
x = base_model8.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model8 = (Model(inputs=base_model8.input, outputs=predictions))


model8.compile()
model8.load_weights('models/first_try.h5', by_name=True)
# model8.save('models/model.h5')

# model = load_model('models/model.h5')
# model.make_predict_function()

@app.route('/', methods=['GET'])
def home():
    result = ''
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict(img_path):
    

    i = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    i = tf.keras.utils.img_to_array(i)/255.0
    i = i.reshape(1, 224, 224, 3)
    pred = model8.predict(i)

    # result = dic[pred[0]]
    print(pred)
    
    return pred[0][0]

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        model8.load_weights('models/first_try.h5', by_name=True)
        img = request.files['myfile']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict(img_path)
        result = dic[round(p)]
        return render_template("index.html", result = result)

if __name__ == '__main__':
    app.run(debug=True)