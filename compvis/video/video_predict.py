import os
import shutil
import cv2
from compvis.model.model import *
from compvis.preprocess.face_detect_single import face_detect_single
from compvis.api.face_labeling import api_output

def frames_to_label_part_video(main_video_path: str, label_frequency=1, start_point=85, end_point=120):
    # Load the full mp4 video, open in cap object
    cap = cv2.VideoCapture(main_video_path)
    frame_rate_fps = int(round(cap.get(cv2.CAP_PROP_FPS),0))
    n_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT),0))

    # Calculate the start and stop frame
    start_frame = start_point*frame_rate_fps
    end_frame = end_point*frame_rate_fps
    n_frames = end_frame - start_frame

    # Calculate frame label frequency and total frames to label based on video properties
    frame_freq = frame_rate_fps*label_frequency
    total_frames_to_label = int(n_frames/frame_freq)

    # set the start frame to iterate from there
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)


    # create a folder for cropped video output
    current_directory = os.getcwd()
    cropped_video_path = os.path.join(current_directory, 'video')
    if os.path.exists(cropped_video_path):
        shutil.rmtree(cropped_video_path, ignore_errors=True)
    os.makedirs(cropped_video_path)

    # Create a VideoWriter object to save the video
    cropped_video_file_path = os.path.join(cropped_video_path,"output_cropped.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(cropped_video_file_path, fourcc, frame_rate_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Create an empty list to store frames to be labeled
    frame_label_list = []

    # Loop through the video object
    for frame in range(start_frame, end_frame):
        ret, img = cap.read()
        # When you reach the end of the movie ret becomes False, it breaks
        if not ret:
            break
        # Frames to be recovered are the multiples of the frame_freq
        if frame % frame_freq == 0:
            # Convert the image to RGB and add it to the list
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_label_list.append(image)
        # Write the current frame to the output video
        video_out.write(img)

     # Report the setting and outputs
    print(f"Movie frame rate: {frame_rate_fps}fps")
    print(f"Movie range: time={start_point}s, frame={start_frame} to time={end_point}s, frame={end_frame} ")
    print(f"Movie total frames: {n_frames}")
    print(f"Choosen detection frequency: every {frame_freq} frames")
    print(f"Total images the analyse: {total_frames_to_label}")

    # Release the video object and the VideoWriter object
    cap.release()
    video_out.release()

    # Return a list with the images of the selected frames, could also return a dictionary with chosen setting outcomes
    return frame_label_list, frame_freq,cropped_video_file_path

def predict_video_frames(frame_label_list,frame_freq,model):

    # For each frame on the list of frames to label, run crop and prediction
    img_labelled_list = []

    for i, img in enumerate(frame_label_list):

        # Crop faces and get the crop path, image and square coordinates
        cropped_img_path, faces_coords, image = face_detect_single(img, 'jpg')

        # predict the faces and store the predicted labels in a list
        label, images = model_predict(model, cropped_img_path , class_names=class_names, target_size=image_size)

        # label the faces
        image_labeled = api_output(image,faces_coords,label)

        # Store labelled frames
        img_labelled_list.append(image_labeled)

    img_labelled_list = [(i*frame_freq, image) for i, image in enumerate(img_labelled_list)]

    return img_labelled_list

# Check if this approach using tuples works, to make things simpler you could just recreate the image list here using the frame_freq and the image_labelled list (i*frame_freq, image)

def make_labeled_video(cropped_video_file_path: str, img_labelled_list: list, label_duration=0.3):

    # Load the original video
    cap = cv2.VideoCapture(cropped_video_file_path)
    frame_rate_fps = int(round(cap.get(cv2.CAP_PROP_FPS),0))
    n_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT),0))

    labled_video_path = os.path.join(os.path.dirname(cropped_video_file_path),"output_video.mp4")
    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(labled_video_path, fourcc, frame_rate_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Set the start frame to iterate from there
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize label frame index
    label_index = 0

    # Calculate the duration of the labelled frames in frames
    label_duration_frames = int(label_duration * frame_rate_fps)

    # Loop through the video and inserted the labelled frames at the original postions
    for i in range(n_frames):
        ret, img = cap.read()

        # Turns false at the end of the object
        if ret == False:
            break

        # If the current frame is a label frame, add and replicate for the chosen duration
        if label_index < len(img_labelled_list) and i == img_labelled_list[label_index][0]:
            label_frame = img_labelled_list[label_index][1]
            # Replicate the labelled frame to make it more obvious
            for j in range(label_duration_frames):
                video_out.write(cv2.cvtColor(label_frame, cv2.COLOR_BGR2RGB)) #Somehow the inserted frames were in BGR and others in RGB
            label_index += 1
        # Otherwise just add the original frame
        else:
            video_out.write(img)
    # os.system("ffmpeg -i {labled_video_path} -vcodec libx264 {labled_video_path}")
    # Release the video objects and the VideoWriter object
    cap.release()
    video_out.release()
    return labled_video_path
