import base64
import io
import json
import string
import time
import os
from unicodedata import category
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import model
import categories
import csv

def prepare_image(img):
    img = Image.open(img)
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    files = json.loads(request.get_data())
    
    if 'file' not in files:
        return "Please try again. The Image doesn't exist"
    
    file = files['file']
    # print("file")

    if not file:
        return


    # or, more concisely using with statement
    with open("imageToSave.jpg", "wb") as fh:
        fh.write(bytes(file, 'utf-8'))
        
        
    img = prepare_image('imageToSave.jpg')
    fruit = files['fruit']
    

    disease, prediction=model.predict(img, fruit)
    for i in categories.category.values():
        for j in i[1].values():
            
            # if j==prediction and j not in (3,4,6,10,14,16,18,20,23,24,25,28):
                
            if disease != "Healthy" and j == prediction:
                with open('disease.csv', 'r') as f:
                    mycsv = list(csv.reader(f))
                    return jsonify(disease=disease, solution=mycsv[j][1])
                
            elif j==prediction:
                return jsonify(disease="Healthy")
            
    
    return jsonify(disease="No Prediction")

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')