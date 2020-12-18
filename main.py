from fastapi import FastAPI
# from services import api
from pydantic import BaseModel
from PIL import Image
import base64,io,uvicorn

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

@app.post('/predict')
def predict(input : Input):
    img = get_img(input.encoded_img)
    # caption=api.predict_caption(img)
    # return {"caption":caption}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
