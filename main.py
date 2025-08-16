from fastapi import FastAPI
import base64
from PIL import Image
import io
from infer import inference
from pydantic import BaseModel

app=FastAPI()
class RequestData(BaseModel):
   img_data:str
@app.post("/predict/")
async def predict(req:RequestData):
   img_data=io.BytesIO(base64.b64decode(req.img_data))
   img=Image.open(img_data)
   pred=inference(img)
   return {"pred":pred}