import os
import io

import hashlib
from PIL import Image
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form

from app.model import predict_class
from app.utils import get_hash
from app.database import DB, DB_CONFIG

db = DB(DB_CONFIG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.create_pool()
    app.state.db = db
    yield
    await db.close_pool()


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def save_prediction(contents: bytes, filename: str, result: str, hash_id: str):
    os.makedirs("uploads", exist_ok=True)
    # если нужно сохранить файл, раскомментируй
    # async with aiofiles.open(filename, "wb") as f:
    #     await f.write(contents)

    await db.save_prediction(hash_id, datetime.utcnow(), filename, result)


@app.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = predict_class(image)

    file_root, file_ext = os.path.splitext(file.filename)
    hash_id = get_hash(file.filename)
    new_filename = f"{hash_id}{file_ext}"
    save_path = f"uploads/{new_filename}"

    if background_tasks:
        background_tasks.add_task(save_prediction, contents, save_path, result, hash_id)

    return {"class": result, "hash_id": hash_id}


@app.post("/save_answer")
async def save_answer(
    prediction_hash: str = Form(...),
    user_answer: str = Form(...)
):
    await db.save_user_answer(
        prediction_hash=prediction_hash,
        user_answer=user_answer,
        answered_at=datetime.utcnow()
    )
    return {"status": "success", "hash": prediction_hash}


