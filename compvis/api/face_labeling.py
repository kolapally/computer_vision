import cv2
color_map = {
    "Angela": (0, 0, 255),
    "Dwight": (0, 0, 100),
    "Jim": (255, 0, 0),
    "Kevin": (0, 0, 150),
    "Michael": (200, 0, 0),
    "Pam": (0, 255, 0),
#     "other person": (200, 0, 0),
#     "trailer": (0, 150, 150),
#     "motorcycle": (0, 150, 0),
#     "bus": (0, 0, 100),
}
def api_output(image,faces_coords,label):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    font = cv2.FONT_HERSHEY_TRIPLEX
    for face,coords in faces_coords.items():
            x, y, w, h = coords
            # draw rectangle
            color = color_map[label[face]]
            stroke = 3
            cv2.rectangle(image, (x, y), (x + w, y + h), color, stroke)
            cv2.putText(image,label[face],(x-5, y-15),font,0.5,color=color)
    return image
