import cv2
color_map = {
    "Angela": (255, 0, 0), # red
    "Dwight": (0, 0, 255), #blue
    "Jim": (0, 255, 0), #green
    "Kevin": (255, 255, 0), #yellow
    "Michael": (128, 0, 128), #purpule
    "Pam": (255, 192, 203), # pink
     "unknown": (255, 165, 0), #orange
}
def api_output(image,faces_coords,label):
    """
    Labels each detected face in an image with their predicted expression.

    Parameters:
    image (numpy.ndarray): The image with detected faces.
    faces_coords (dict): A dictionary containing the coordinates of the detected faces.
    label (dict): A dictionary containing the predicted expression labels for each detected face.

    Returns:
    numpy.ndarray: The input image with labeled faces.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for face,coords in faces_coords.items():
            x, y, w, h = coords
            # draw rectangle
            color = color_map[label[face]]
            stroke = 3
            cv2.rectangle(image, (x, y), (x + w, y + h), color, stroke)
            cv2.putText(image,label[face],(x-5, y-15),font,0.9,color=color)
    return image
