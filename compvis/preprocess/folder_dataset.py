from  tensorflow.keras.utils import image_dataset_from_directory
from compvis.params import *

def folder_to_dataset_splits(folder_path, labels='inferred',
                              label_mode='categorical',
                              image_size=image_size,
                              validation_split = 0.2,
                              batch_size=batch_size,
                              interpolation = 'bicubic',
                              crop_to_aspect_ratio = True,
                              train_split = 0.2
                              ):
     """
    Creates train, validation, and test datasets from a folder of images.

    Args:
        folder_path (str): Path to the folder containing the images.
        labels (str, optional): Type of labels to use for the dataset. Can be 'inferred' to use the folder structure, or a path to a CSV file containing labels. Defaults to 'inferred'.
        label_mode (str, optional): Type of label encoding to use. Can be 'categorical' for one-hot encoding, 'binary' for binary encoding, 'sparse' for integer encoding, or None for no labels. Defaults to 'categorical'.
        image_size (tuple, optional): Size to which the input images should be resized. Defaults to the value of the image_size constant from the compvis.params module.
        validation_split (float, optional): Fraction of the data to reserve for validation. Defaults to 0.2.
        batch_size (int, optional): Size of the batches for the dataset. Defaults to the value of the batch_size constant from the compvis.params module.
        interpolation (str, optional): Interpolation method to use when resizing the images. Can be 'nearest', 'bilinear', 'bicubic', or 'lanczos3'. Defaults to 'bicubic'.
        crop_to_aspect_ratio (bool, optional): Whether to crop the images to the specified aspect ratio. Defaults to True.
        train_split (float, optional): Fraction of the training data to use for training. The rest is used for validation. Defaults to 0.2.

    Returns:
        tuple: A tuple containing the training, validation, and test sets, as well as the class names.

    Raises:
        ValueError: If validation_split is not between 0 and 1, or if train_split is not between 0 and 1.

    """
    train, test_set = image_dataset_from_directory(
                                                directory = folder_path,
                                                labels = labels,
                                                label_mode =label_mode,
                                                image_size = image_size,
                                                batch_size=batch_size,
                                                validation_split = validation_split,
                                                seed = 0,
                                                subset = 'both',
                                                interpolation = interpolation,
                                                crop_to_aspect_ratio = crop_to_aspect_ratio
                                                )

    train_batches = int((1-train_split)*train.cardinality().numpy())
    class_names = train.class_names

    train_set = train.take(train_batches)
    val_set = train.skip(train_batches)

    return train_set, val_set, test_set, class_names

if __name__ == "__main__":
    folder_to_dataset_splits()
