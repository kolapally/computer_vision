from mtcnn import MTCNN
import cv2
import os
from glob import glob
import shutil

def face_detect_multiple (raw_img_path: str) -> None:
    '''
    Detects faces in a a series of folders to create the cropped faces dataset
    Need to finish the saving part
    '''
    # Get image folder list
    folder_list = glob(os.path.join(raw_img_path, '*'))

    # Create a list of folders and filenames -> getting only png for now
    filename_list = [glob(os.path.join(folder, "*.png")) for folder in folder_list]

    detector = MTCNN()
    for img_path in filename_list:
        for i, img in enumerate(img_path):
            image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
            #Outputs a faces list of dict, with the bounding box inside the key 'box'
            faces = detector.detect_faces(image)
            #run crop faces function
            crop_faces(img, image, faces)
    return None

def crop_faces(img, image, faces):
    if len(faces) > 0:
        for i, face in enumerate(faces):

            # Create the directory to save crops if it does not exist
            current_directory = os.getcwd()
            cropped_img_path = os.path.join(current_directory, 'cropped')
            if not os.path.exists(cropped_img_path):
                os.makedirs(cropped_img_path)

            # Get face box coordinates
            x, y, w, h = face['box']

            # get face and save crop
            face = image[y:y + h, x:x + w]
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{img[:-4].replace('Kaggle_clean', 'cropped')}_face{i}.png", face)

            print(f"{img[:-4].replace('Kaggle_clean', 'cropped')}_face{i}.png is saved")
    return None
