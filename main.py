from fastapi import FastAPI
# from services import api
from pydantic import BaseModel
from PIL import Image
import base64,io,uvicorn

import numpy as np
import re, json,pickle ,keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

app =FastAPI()

class Input(BaseModel):
    encoded_img : str

@app.get('/')
def index():
    return "Hello people, This is an API developed by Harshit Muhal for image captioning. Thanks for visiting :) :)"

def get_img(encoded_img):
    encoded_img_bytes = encoded_img.encode('utf-8')
    base64bytes = base64.b64decode(encoded_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    img = img.resize((224,224))
    return img

def predict_caption(img):
    # Resnet model
    model = ResNet50(weights="imagenet",input_shape=(224,224,3))
    new_model = Model(model.input,model.layers[-2].output)
    # vocabulary
    with open("services/word_to_idx.pkl","rb") as f:
        word_to_idx=pickle.load(f)
    with open("services/idx_to_word.pkl","rb") as f:
        idx_to_word=pickle.load(f)
    # Main Model
    json_file = open('services/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('services/model_weights/model_final.h5')

    # preprocessing
    img = image.img_to_array(img)
    img = img.reshape((1,224,224,3))
    img = preprocess_input(img)
    # encoding images
    feature_vector = new_model.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    photo=feature_vector.reshape((1,2048))
    # Finding caption
    in_text = '<start>'
    max_len = 37
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len,padding='post')
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word[yhat]
        in_text += (' ' + word)
        if word == '<end>':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

@app.post('/predict')
def predict(input : Input):
    img = get_img(input.encoded_img)
    caption=predict_caption(img)
    return {"caption":caption}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
