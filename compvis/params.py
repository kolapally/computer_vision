import os
class_names = ['Angela','Dwight','Jim','Kevin','Michael','Pam','unknown']
#path to model .h5 that we save or get it from cloud
image_size = (128,128)
image_shape = (128,128,3)
model_path = '../../models/sgd-adam.h5'
threshold = 0.7
batch_size = 32
epochs = 100
patience = 10
bucket_name = os.environ.get('BUCKET_NAME')
