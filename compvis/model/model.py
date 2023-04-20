# All needed libraries are imported for model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
# import efficientnet.keras as efn
from tensorflow.keras.applications import ResNet152
#from tensorflow.keras.optimizers.experimental import SGD
#from tensorflow.keras.activations import gelu
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from compvis.params import *
from google.cloud import storage
import numpy as np
import os
import time

def model_init(image_shape):
    # crop/Input shape should be a global variable to be the same everywhere
     """
    Initializes a pre-trained model (transfer learning) model for image classification.
    
    Args:
    image_shape: tuple, the shape of the input image (height, width, channels(RGB)).
    
    Returns:
    model: a tenserflow.Keras Sequential model object.
    """
    base_model = ResNet152(input_shape = image_shape, include_top = False, weights = 'imagenet')

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Create a new model on top of the pre-trained VGG16 model
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(130, activation='relu'),
        Dense(6, activation='softmax')
        ])
    print("✅model initialized")
    print(model.summary())
    return model

def model_compile(model,
               loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy']):
    # optimizer = optimizers.Adam(learning_rate=learning_rate)
    """
    Compiles the specified tenserflow.Keras model with the given loss, optimizer, and evaluation metrics.
    
    Args:
    model: a Keras model object.
    loss: string, the loss function to use for training the model.
    optimizer: string or optimizer object, the optimizer to use for training the model.
    metrics: list of strings, the evaluation metrics to use for the model.
    
    Returns:
    model: the compiled tenserflow.Keras model object.
    """
    model.compile(loss=loss,
              optimizer = optimizer,
              metrics = metrics
             )
    print("✅ model compiled")
    return model

def model_train(model, train_set, val_set, epochs = epochs, patience = patience):
    
    """
    Trains the specified tenserflow.Keras model on the given training and validation sets.
    
    Args:
    model: a tenserflow.Keras model object.
    train_set: a tenserflow.Keras ImageDataGenerator object for the training set.
    val_set: a tenserflow.Keras ImageDataGenerator object for the validation set.
    epochs: integer, the number of epochs to train the model for.
    patience: integer, the number of epochs to wait before early stopping if the validation accuracy does not improve.
    
    Returns:
    model: the trained tenserflow.Keras model object.
    history: a tenserflow.Keras History object containing the training history.
    """
    es = EarlyStopping(monitor='val_accuracy', patience = patience, restore_best_weights=True)

    history = model.fit(train_set,
                  epochs = epochs,
                  batch_size=32,
                  verbose=1,
                  validation_data=val_set,
                  callbacks = [es]
                  )
    print("✅ model trained")

    return model, history

def model_eval(model, test_set) -> None:
    """
    Evaluates the specified tenserflow.Keras model on the given test set.
    
    Args:
    model: a tenserflow.Keras model object.
    test_set: a tenserflow.Keras ImageDataGenerator object for the test set.
    
    Returns:
    None
    """
    loss, accuracy = model.evaluate(test_set)
    print(f"✅ Model loss:{loss:.2f}, accuracy:{accuracy:.2f}")
    return None

def model_predict(model, cropped_img_path, class_names, target_size=image_size,threshold=threshold):

    # The target_size should be a global variable to use in all
    """
    Uses the specified Keras model to make predictions on the images in the given directory.
    
    Args:
    model: a Keras model object.
    cropped_img_path: string, the path to the directory containing the images to predict on.
    class_names: list of strings, the names of the classes that the model can predict.
    target_size: tuple, the size of the input images to the model (height, width).
    threshold: float, the probability threshold below which predictions will be considered "unknown".
    
    Returns:
    label: list of strings, the predicted labels for each of the input images.
    images: list of PIL Image objects, the input images that were predicted on.
    """
    
    img_paths = sorted (os.listdir(cropped_img_path))

    label = []
    images = []
    for img_path in img_paths:
        img = image.load_img(os.path.join(cropped_img_path, img_path),
                             target_size=target_size,
                             interpolation='bicubic',
                             keep_aspect_ratio=True)

        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)

        prediction = model.predict(img_preprocessed)
        if np.max(prediction,axis=1)[0] > threshold:
            label.append(class_names[np.argmax(prediction,axis=1)[0]])
        else:
            label.append(class_names[len(class_names)-1])

        images.append(img)
        print(f'{img_path}--{np.max(prediction,axis=1)[0]}--{class_names[np.argmax(prediction,axis=1)[0]]}')
    print("✅ Prediction completed")

    return label, images

def model_save (model, model_name:str):
    """
    Saves the specified tenserflow.Keras model to a local file with the given name and timestamp.
    
    Args:
    model: a tenserflow.Keras model object.
    model_name: string, the name to give the saved model file.
    
    Returns:
    None
    """
    
    # Add a local path to save this
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save model in the same location inside models folder
    current_directory = os.getcwd()
    save_folder = os.path.join(current_directory, "models")
    os.makedirs(save_folder, exist_ok=True)

    model_path = os.path.join(save_folder, f"{timestamp}_{model_name}.h5")

    model.save(model_path)

    print("✅ Model saved locally")

def model_load(model_path):
    """
    Loads the specified tenserflow.Keras model from a local file or from a Google Cloud Storage bucket.
    
    Args:
    model_path: string, the path to the model file to load.
    
    Returns:
    model: a tenserflow.Keras model object, or None if the model could not be loaded.
    """
    # Load from local if it exists
    current_path = os.path.dirname(__file__)
    model_path_final = os.path.join(current_path, model_path)
    if os.path.exists(model_path_final):
        print("loading from local disk")
        model = load_model(model_path_final, compile=False)
        print(" ✅ Model loaded from local disk")
        return model
    # Load from cloud if it does not exist
    else:
        print("loading from cloud")
        client = storage.Client()
        blobs = list(client.get_bucket(bucket_name).list_blobs(prefix="models"))
        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            model_path_to_save = latest_blob.name
            latest_blob.download_to_filename(model_path_to_save)
            model = load_model(model_path_to_save)
            print("✅ Latest model downloaded from cloud storage")
            return model
        except:
            print(f"\n:x: No model found on GCS bucket {bucket_name}")
            return None
