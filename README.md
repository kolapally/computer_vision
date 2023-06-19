# CompVis - Computer Vision for Industrial Safety
Welcome to CompVis, our project in implementing a Deep Learning model to detect and recognize faces from video files applied to industrial safety.

# 💡 Project Developers
<table>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/121227950/232547987-8162ccde-d3ba-4893-bcdf-fbe1c1b470f5.png" width="80"></td>
    <td>Kolapally Sai kalyan</td>
    <td><a href="https://github.com/kolapally" target="_blank">GitHub</a></td>
    <td><a href="https://troopl.com/kolapally" target="_blank">Portfolio</a></td>
    <td><a href="https://www.linkedin.com/in/kolapally/" target="_blank">LinkedIn</a></td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/121227950/232478965-e247fd12-5c77-44ee-a97f-a977737a070f.png" width="80"></td>
    <td>Daniel Osório</td>
    <td><a href="https://github.com/dosorio79" target="_blank">GitHub</a></td>
    <td><a href="https://troopl.com/danielsosorio" target="_blank">Portfolio</a></td>
    <td><a href="https://www.linkedin.com/in/dosorio/" target="_blank">LinkedIn</a></td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/121227950/232478686-028b704f-3a58-4c9c-894e-b1a0e10ea8f3.png" width="80"></td>
    <td>Merle Buchmann</td>
    <td><a href="https://github.com/marierae">GitHub</a></td>
    <td>Portfolio</td>
    <td><a href="https://www.linkedin.com/in/merle-buchmann-784285163/">LinkedIn</a></td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/121227950/232479017-85aaa550-1856-47e6-a56c-faf54daa21fd.png" width="80"></td>
    <td>Kranthi Maddishetty</td>
    <td>GitHub</td>
    <td>Portfolio</td>
    <td><a href="https://www.linkedin.com/in/kranthi-maddishetty/">LinkedIn</a></td>
  </tr>
</table>

# 🔭 Project Overview
The primary objective of our project was to **leverage computer vision and advanced face detection and recognition technology for emergency monitoring** in industrial safety contexts. To achieve this goal, we used an episode of the popular TV show "The Office" that depicted a fire drill scenario. We then trained our face recognition model using a dataset of six characters from the show. Our ultimate aim is to **create a robust and reliable tool that can help improve emergency response** by quickly identifying and tracking individuals.

# 🖧 Tech Stack
- Python backend:
  - Image and video manipulation using <a href="https://opencv.org/">OpenCV</a>
  - Face detection using <a href="https://pypi.org/project/mtcnn/">MTCNN</a>
  - Face recognition deep learning model using <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications">TensorFlow transfer learning</a> with EfficientNetV2 trained with a dataset of 1500 images of each character after image augmentation
- API with <a href="https://fastapi.tiangolo.com/">FastAPI</a>
- <a href="https://streamlit.io/">Streamlit</a> frontend

# 📌 App tutorial
 🧪 You can test our app here https://compvis.streamlit.app/ 🧪
<table>
  <tr>
    <td><b>Detection and identification of faces on a single image</b>
    <ul>-Select the image option on the left navigation bar, upload an image and the app returns a final image with a face bounding box and predicted face identification</ul>
        <ul>-By default the app uses a threshold of 70% probability to consider the identification positive. Bellow that value faces are labeled as unknown</ul>
    </td>
    <td><img src="https://user-images.githubusercontent.com/121227950/232462331-08a0adcd-d1a1-4f39-89a9-bfb0f2e53509.png" width="500"> 
      (Click to enlarge)
    </td>
  </tr>
  <tr>
    <td><b>Detection and identification of faces on a video</b>
    <ul>-Select the video option on the left navigation bar, upload a video, and the app returns a final video with a face bounding box and predicted face identification</ul>
    <ul>-By default the app will label the first 30 seconds of the video and will sample faces every second. The app returns a video with a face bounding box and predicted face identification. The labeled frames are duplicated a defined number of times to better visualize them</ul>
    </td>
    <td><img src="https://user-images.githubusercontent.com/121227950/232471237-8ac04bc3-749a-4ab7-9f50-db691895acf2.png" width="500"> (Click to enlarge)</td>
  </tr>  
   <tr>
    <td style="vertical-align:top"><b>See a quick demo of this app <a href=https://drive.google.com/file/d/1nbOl3mUyAyEcyQH7v04KyAABqTwXVLKO/view?usp=drive_link">here</a></b></td>
    <td>
      <img src="https://github.com/kolapally/computer_vision/assets/121227950/383a339d-5dab-4579-90be-9f29daae4a0b">
    </td> 
  </tr>  
</table>

# 🚀Project scope and duration
This project was developed as part of the Le Wagon Data Science Bootcamp Batch 1181 Online (Feb-Mar2023) over two weeks
