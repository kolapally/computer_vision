from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from compvis.preprocess.face_detect_single import face_detect_single
from compvis.model.model import *
from compvis.api.face_labeling import api_output
from compvis.params import *
import numpy as np
import cv2
app = FastAPI()
app.state.model = model_load(model_path)
# Allow all requests (optional, good for development purposes)
app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],  # Allows all origins
     allow_credentials=True,
     allow_methods=["*"],  # Allows all methods
     allow_headers=["*"],  # Allows all headers
 )

@app.get("/")
def index():
    return {"status": "ok"}

@app.post("/detect_faces")
async def detect_faces(img: UploadFile=File(...)):
    # Receiving and decoding the image
    contents = await img.read()

    # Convert bytes to numpy array and then to cv2 object
    np_array = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Call face_detect_single function and get results
    cropped_img_path, faces_coords, image = face_detect_single(img,'jpg')

    # Other elements of the pipeline will go here:
    # Crop -> Predict -> get classified img
    #load trained model
    # model = model_load('/home/kolapally/code/kolapally/computer_vision/compvis/model/models/model.h5')
    #Office cast name lables

    #predict the input image
    label, images = model_predict(app.state.model,cropped_img_path , class_names, target_size=image_size)
    image_output = api_output(image,faces_coords,label)
    # Encoding and responding with the image
    im = cv2.imencode('.png', image_output)[1] # extension depends on which format is sent from Streamlit
    return Response(content=im.tobytes(), media_type="image/png")
