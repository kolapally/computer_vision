from mtcnn import MTCNN
import cv2
import os
import shutil

def face_detect_single(image:str, file_type:str)-> str:
    '''
    Detects faces in a single image, used for inputs to be predicted. Saves
    the cropped faces in a cropped directory that is erased everytime you run
    the function so that the faces are not mixed. The coordinates of each box
    are saved in a dictionary that can be used to plot them after predict
    '''

    # Clean cropped directory if it exists
    current_directory = os.getcwd()
    cropped_img_path = os.path.join(current_directory, 'cropped')
    if os.path.exists(cropped_img_path):
        shutil.rmtree(cropped_img_path, ignore_errors=True)

    # Load image as CV2 object
    image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

    #Outputs a faces list of dict, with the bounding box inside the key 'box'
    detector = MTCNN()
    faces = detector.detect_faces(image)

    #Loop through the faces, save box coordinates and save each of them in a face_crop folder
    faces_coords = {}

    if len(faces) > 0:
        for i, face in enumerate(faces):
            x, y, w, h = face['box']

            # Store coordinates of each face in a dict
            faces_coords[f'face{i}'] = (x, y, w, h)

            # get face crop and make it RGB
            face = image[y:y + h, x:x + w]
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

            # create cropped folder in the current directory
            os.makedirs(cropped_img_path)

            # Write image in cropped directory
            file_path = f"{cropped_img_path}/image_face{i}.{file_type}"
            cv2.imwrite(file_path, face)
            print(f"{file_path} is saved")

    return cropped_img_path
