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
import numpy as np
import os
import time

def model_init(input_shape):
    # crop/Input shape should be a global variable to be the same everywhere

    base_model = ResNet152(input_shape = input_shape, include_top = False, weights = 'imagenet')

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

    model.compile(loss=loss,
              optimizer = optimizer,
              metrics = metrics
             )
    print("✅ model compiled")
    return model

def model_train(model, train_set, val_set, epochs = 100, patience = 10):

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
    loss, accuracy = model.evaluate(test_set)
    print(f"Model loss:{loss}, accuracy:{accuracy}")
    return None

def model_predict(model, cropped_img_path, class_names, target_size=(96,96)):

    # The target_size should be a global variable to use in all

    img_paths = os.listdir(cropped_img_path)

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
        label.append(class_names[np.argmax(prediction,axis=1)[0]])
        images.append(img)

    print("✅ Prediction completed")

    return label, images

def model_save (model, model_name:str):
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
    # Add a local path to load this
    model = load_model(model_path)
    print("✅ Model loaded from local disk")
    return model
