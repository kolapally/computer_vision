from mtcnn import MTCNN
import cv2
import os
import shutil

def face_detect_single(image):
    '''
    Detects faces in a single image, used for inputs to be predicted. Saves
    the cropped faces in a cropped directory that is erased everytime you run
    the function so that the faces are not mixed. The coordinates of each box
    are saved in a dictionary that can be used to plot them after predict.
    V2 gets a CV2 image object directly from API.
    '''

    # Clean cropped directory if it exists
    current_directory = os.getcwd()
    cropped_img_path = os.path.join(current_directory, 'cropped')
    if os.path.exists(cropped_img_path):
        shutil.rmtree(cropped_img_path, ignore_errors=True)
    os.makedirs(cropped_img_path)

    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Outputs a faces list of dict, with the bounding box inside the key 'box'
    detector = MTCNN()
    faces = detector.detect_faces(image)

    #Loop through the faces, save box coordinates and save each of them in a face_crop folder
    faces_coords = {}
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if len(faces) > 0:
        for i, face in enumerate(faces):
            x, y, w, h = face['box']

            # Store coordinates of each face in a dict
            faces_coords[f'face{i}'] = (x, y, w, h)

            # get face crop and make it RGB
            face = image[y:y + h, x:x + w]
            # face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

            # Write image in cropped directory
            file_path = f"{cropped_img_path}/image_face{i}.png"
            cv2.imwrite(file_path, face)
            print(f"{file_path} is saved")

            # draw rectangle
            color = (0, 255, 255) # in BGR
            stroke = 3
            cv2.rectangle(image, (x, y), (x + w, y + h), color, stroke)

    return cropped_img_path, faces_coords, image

if __name__ == "__main__":
    face_detect_single()
