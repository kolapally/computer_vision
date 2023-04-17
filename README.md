# CompVis - Computer Vision for Industrial Safety
Welcome to CompVis, our proeject in implementing a Deep Learning model to detect and recognize faces from video files applied to industrial safety.

# 💡 Project Developers
<table>
  <tr>
    <td>Kolapally Sai kalyan</td>
    <td><a href="https://github.com/kolapally" target="_blank">GitHub</a></td>
    <td>Portfolio</td>
    <td><a href="https://www.linkedin.com/in/kolapally/" target="_blank">LinkedIn</a></td>
  </tr>
  <tr>
    <td>Daniel Osório</td>
    <td><a href="https://github.com/dosorio79" target="_blank">GitHub</a></td>
    <td><a href="https://troopl.com/danielsosorio" target="_blank">Portfolio</a></td>
    <td><a href="https://www.linkedin.com/in/dosorio/" target="_blank">LinkedIn</a></td>
  </tr>
  <tr>
    <td>Merle Buchmann</td>
    <td>GitHub</td>
    <td>Portfolio</td>
    <td>LinkedIn</td>
  </tr>
  <tr>
    <td>Kranthi Maddishetty</td>
    <td>GitHub</td>
    <td>Portfolio</td>
    <td>LinkedIn</td>
  </tr>
</table>

# 🔭 Project overview
The primary objective of our project was to leverage computer vision and advanced face detection and recognition technology for emergency monitoring in industrial safety contexts. To achieve this goal, we used an episode of the popular TV show "The Office" that depicted a fire drill scenario. We then trained our face recognition model using a dataset of six characters from the show. Our ultimate aim is to create a robust and reliable tool that can help improve emergency response in industrial settings by quickly identifying and tracking individuals during emergency situations.

# 🖧 Tech Stack
- Python backend - Face detection using <a href="https://pypi.org/project/mtcnn/">MTCNN</a>, face recognition deep learning model using <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications">TensorFlow transfer learning</a> with EfficientNetV2 trained with a dataset of 1500 images of each character after image augmentation.
- API with <a href="https://fastapi.tiangolo.com/">FastAPI</a>
- <a href="https://streamlit.io/">Streamlit</a> frontend

# 📌 App tutorial
<table>
  <tr>
    <td><b>Detection and identification of faces on a single image</b>
    <ul>-Select the image option on the left navigation bar, upload an image and the app returns a final image with a face bounding box and predicted face identification</ul>
        <ul>-By default the app uses a threshold of 70% probability to consider the identification positive. Bellow that value faces are labelled as unknown</ul>
    </td>
    <td><img src="https://user-images.githubusercontent.com/121227950/232462331-08a0adcd-d1a1-4f39-89a9-bfb0f2e53509.png" width="500"></td>
  </tr>
  <tr>
    <td><b>Detection and identification of faces on a video</b>
    <ul>-Select the video option on the left navigation bar, upload an video and the app returns a final video with a face bounding box and predicted face identification</ul>
    <ul>-By default the app will label the first 30 seconds of the video and will sample for faces every second. The app returns a video with face bounding box and predicted face identification. The labelled frames are duplicated for a defined number of times in order to better visualise them</ul>
    </td>
    <td><img src="https://user-images.githubusercontent.com/121227950/232462331-08a0adcd-d1a1-4f39-89a9-bfb0f2e53509.png" width="500"></td>
  </tr>  
</table>

# 🚀Project scope and duration
This project was developed as part of the Le Wagon Data Science Bootcamp Batch 1181 Online (Feb-Mar2023) over the course of two weeks
