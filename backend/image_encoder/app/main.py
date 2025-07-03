from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import os
import numpy as np
import gdown

url = "https://drive.google.com/drive/folders/1z5E0u_dPoZGU9tM7TIZbpJQV1dFgE0Bk"
model_path = "weights/image_encoder.pth"

app = FastAPI()


class ImageModel:
    def __init__(self):
        self.model = None

    def load_weights(self, url: str, model_path: str):
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            gdown.download(url, model_path, quiet=False)
        # self.model =


    def encode(self, image: Image.Image) -> list:
        return np.ones(128).tolist()


model = ImageModel()
model.load_weights(url, model_path)


@app.post("/encode_image")
async def encode_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    vector = model.encode(image)
    return {"vector": vector}
