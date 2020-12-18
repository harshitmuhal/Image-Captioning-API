import numpy as np
import re, json,pickle ,keras
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model, model_from_json
from keras.preprocessing.sequence import pad_sequences
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = ResNet50(weights="imagenet",input_shape=(224,224,3))
new_model = Model(model.input,model.layers[-2].output)

def get_image_encodings(img):
    feature_vector = new_model.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector

def preprocessing(img):
    # img = image.load_img(img_path,target_size=(224,224))
    img = image.img_to_array(img)
    img = img.reshape((1,224,224,3))
    img = preprocess_input(img)
    return get_image_encodings(img)

with open("services/word_to_idx.pkl","rb") as f:
    word_to_idx=pickle.load(f)
with open("services/idx_to_word.pkl","rb") as f:
    idx_to_word=pickle.load(f)

max_len = 37

json_file = open('services/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('services/model_weights/model_final.h5')

def predict_caption(img_path):
    photo=preprocessing(img_path)
    photo=photo.reshape((1,2048))
    in_text = '<start>'
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
