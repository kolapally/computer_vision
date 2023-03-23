from  tensorflow.keras.utils import image_dataset_from_directory

def folder_to_dataset_splits(folder_path, labels='inferred',
                              label_mode='categorical',
                              image_size=(128,128),
                              validation_split = 0.2,
                              batch_size=32,
                              interpolation = 'bicubic',
                              crop_to_aspect_ratio = True,
                              train_split = 0.2
                              ):

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
    train_set = train.take(train_batches)
    val_set = train.skip(train_batches)

    class_names = train_set.class_names

    return train_set, val_set, test_set, class_names

if __name__ == "__main__":
    folder_to_dataset_splits()
