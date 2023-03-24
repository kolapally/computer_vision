from mtcnn import MTCNN
import cv2
import os
from glob import glob

def face_detect_multipleV2(raw_img_path: str, file_type: str) -> str:
    '''
    Detects faces in a a series of folders to create the cropped faces dataset
    Takes a file path and the file extension of the image files (png or jpg...)
    Saves the files inside a cropped dir in the raw image path, inside a folder
    with the same name as the original img/pam -> img/cropped/Pam
    '''
    # Get image folder list
    folder_list = glob(os.path.join(raw_img_path, '*'))

    # Create the crop dir
    cropped_folder = os.path.join(raw_img_path, 'cropped')
    os.makedirs(cropped_folder, exist_ok=True)

    for folder in folder_list:
        # Create subfolder for cropped faces
        cropped_subfolder = os.path.join(cropped_folder, os.path.basename(folder))
        os.makedirs(cropped_subfolder, exist_ok=True)

        # Get image files in folder
        img_files = glob(os.path.join(folder, f'*.{file_type}'))

        # Loop over image files and crop faces
        for img_file in img_files:
            # Load image
            image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

            # Detect faces
            detector = MTCNN()
            faces = detector.detect_faces(image)

            if len(faces) > 0:
                # Loop over faces and crop/save
                for i, face in enumerate(faces):
                    # Get face square and convert to color
                    x, y, w, h = face['box']
                    face = image[y:y + h, x:x + w]
                    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

                    # Save cropped face
                    face_name = f"{os.path.splitext(os.path.basename(img_file))[0]}_face{i}.{file_type}"
                    face_path = os.path.join(cropped_subfolder, face_name)
                    cv2.imwrite(face_path, face)
                    print(f"{face_path} is saved")

            else:
                print(f"No faces detected in {img_file}")

    return cropped_folder

if __name__ == "__main__":
    face_detect_multipleV2()
