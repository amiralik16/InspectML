# Inspiered by https://github.com/jabertuhin/image-classification-api/blob/development/app/main.py
# and https://github.com/amitrajitbose/cat-v-dog-classifier-pytorch

from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import sys
import logging
from prometheus_fastapi_instrumentator import Instrumentator

from model.predict import predict_image


class ImageResponse(BaseModel):
    filename: str
    contentype: str    
    predicted_class: str
    probability: float


app = FastAPI()

@app.post("/predict/", response_model=ImageResponse)
async def predict(file: UploadFile = File(...)):
    logging.info(file.content_type)
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')    

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        prediction_dict = predict_image(image)
        
        return {
            "filename": file.filename, 
            "contentype": file.content_type,            
            "predicted_class": prediction_dict['class'],
            "probability": prediction_dict['confidence']
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def hello():
    return {'send image to /predict to classify cat vs dog'}

Instrumentator().instrument(app).expose(app)
