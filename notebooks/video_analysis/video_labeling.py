import cv2

def video_labelled(video_path, img_labelled_list, label_frequency=1, label_duration=2):

    # Load the original video
    cap = cv2.VideoCapture(video_path)
    frame_rate_fps = int(round(cap.get(cv2.CAP_PROP_FPS),0))
    Label_duration_frames = label_duration * frame_rate_fps

    # Create a VideoWriter object get video properties
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter("output_labeled.mp4", fourcc, frame_rate_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Set the start frame to iterate from there
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize frame counter and label frame index
    frame_count = 0
    label_index = 0

    # Loop through the video and add insert labelled frames at their positions replicating to make them visible
    while True:
        ret, img = cap.read()

        # When you reach the end of the movie ret becomes False, it breaks
        if ret == False:
            break

        # If the current frame is a labeled frame, insert it and replicate, else just keep original
        breakpoint()
        if frame_count % label_frequency == 0:
            label_img = img_labelled_list[label_index]
            # Replicate the labelled image n times based on label_duration
            for i in range(Label_duration_frames):
                video_out.write(label_img)
        else:
            video_out.write(img)

        # Increment the counters
        frame_count += 1
        if frame_count % label_frequency == 0:
            label_index += 1

    # Release the video objects and the VideoWriter object
    cap.release()
    video_out.release()

    # file should be saved localy so no return, just load it in the API check

    return None
