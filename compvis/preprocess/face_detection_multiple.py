from mtcnn import MTCNN
import cv2
import os
from glob import glob

def face_detect_multiple (raw_img_path: str, file_type:str) -> str:
    '''
    Detects faces in a a series of folders to create the cropped faces dataset
    Takes a file path and the file extension of the image files (png or jpg...)
    Need to finish the saving part
    '''
    # Get image folder list
    folder_list = glob(os.path.join(raw_img_path, '*'))

    # Create a list of folders and filenames -> getting only png for now
    filename_list = [glob(os.path.join(folder, f"*.{file_type}")) for folder in folder_list]

    detector = MTCNN()
    for img_path in filename_list:
        for i, img in enumerate(img_path):
            image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
            #Outputs a faces list of dict, with the bounding box inside the key 'box'
            faces = detector.detect_faces(image)
            #run crop faces function
            crop_faces(raw_img_path, file_type, img, image, faces)
            cropped_folder = os.path.join(raw_img_path, 'cropped')

    return cropped_folder

def crop_faces(cropped_folder, file_type, img, image, faces) -> None:
    if len(faces) > 0:

        # Get file name and parent directory
        filename = os.path.basename(img)
        par_dir = os.path.basename(os.path.dirname(img))

        # Create the directory to save crops if it does not exist
        cropped_img_path = os.path.join(cropped_folder, par_dir)
        if not os.path.exists(cropped_img_path):
            os.makedirs(cropped_img_path)

        # Get faces and save crops
        for i, face in enumerate(faces):
            # Get face square convert to color
            x, y, w, h = face['box']
            face = image[y:y + h, x:x + w]
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

            # Save cropped face
            face_name = f"{filename[:-4]}_face{i}.{file_type}"
            face_path = os.path.join(cropped_img_path, face_name)
            cv2.imwrite(face_path, face)
            print(f"{face_path} is saved")

    else:
        print('Error: no faces detected!')
