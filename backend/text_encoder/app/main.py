from fastapi import FastAPI, Form, File, UploadFile
import numpy as np

app = FastAPI()


class DummyTextModel:
    def load_weights(self, path: str):
        print(f"Weights loaded from {path}")

    async def encode_from_file(self, file_content: bytes) -> list:
        return np.zeros(128).tolist()


model = DummyTextModel()
model.load_weights("weights/text.pt")


@app.post("/encode_text")
async def encode_text(file: UploadFile = File(...)):
    contents = await file.read()
    vector = await model.encode_from_file(contents)

    return {"vector": vector}
