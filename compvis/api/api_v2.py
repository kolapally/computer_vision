from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from fastapi.responses import FileResponse
from compvis.preprocess.face_detect_single import face_detect_single
from compvis.model.model import *
from compvis.api.face_labeling import api_output
from compvis.params import model_path,class_names,image_size
from compvis.video.video_predict import *
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

@app.post("/detect_image")
async def detect_image(img: UploadFile=File(...)):
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
    #class_names = ['Angela','Dwight','Jim','Kevin','Michael','Pam','unknown']

    #predict the input image
    label, images = model_predict(app.state.model,cropped_img_path , class_names=class_names, target_size=image_size)
    image_output = api_output(image,faces_coords,label)
    # Encoding and responding with the image
    im = cv2.imencode('.png', image_output)[1] # extension depends on which format is sent from Streamlit
    return Response(content=im.tobytes(), media_type="image/png")

@app.post("/detect_video")
async def detect_video(video: UploadFile=File(...)):
    # Receiving and decoding the image
    # contents = await video.read()
     # Save the video file locally
    with open("input_video.mp4", "wb") as buffer:
        buffer.write(await video.read())

    current_directory = os.getcwd()
    main_video_path = os.path.join(current_directory, 'input_video.mp4')

    # crops the given video and selects the frame to label
    frame_label_list, frame_freq,cropped_video_file_path = frames_to_label_part_video(main_video_path, label_frequency=1, start_point=85, end_point=120)

    # predicts the selected video frames
    img_labelled_list = predict_video_frames(frame_label_list,frame_freq,model = app.state.model)

    # ouputs the video with labled frames
    labled_video_path = make_labeled_video(cropped_video_file_path, img_labelled_list, label_duration=0.3)
    headers = {
        "Content-Disposition": "attachment; filename=output_video.mp4",
        "Content-Type": "video/mp4",
    }

    mp4_file_path = labled_video_path
    with open(mp4_file_path, "rb") as f:
        file_content = f.read()
    # im = cv2.imencode('.png', image_output)[1] # extension depends on which format is sent from Streamlit
    return Response(content=file_content,headers=headers)
