import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


IMG_SIZE = (180, 180)  
BATCH_SIZE = 32         
VAL_SPLIT = 0.2         

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),       
    layers.RandomRotation(0.1),            
    layers.RandomZoom(0.1) # do 10%
])

def load_train_val(data_dir="data/train"):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=42, # da podela u train i val uvek bude ista
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names

    # normalizacija
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True)/255.0, y))
    val_ds = val_ds.map(lambda x, y: (x/255.0, y))

    return train_ds, val_ds, class_names


def load_test(data_dir="data/test"):
   
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False  
    )

    class_names = test_ds.class_names

    # normalizacija 
    test_ds = test_ds.map(lambda x, y: (x/255.0, y))

    return test_ds, class_names


def count_images_per_class(dataset, class_names):
    counts = {class_name: 0 for class_name in class_names} # recnik, "basketball": 0 npr
    
    for images, labels in dataset:
        for label in labels.numpy():
            counts[class_names[label]] += 1
    
    return counts

def show_sample_images(dataset, class_names, samples_per_class=1, save_path="sample_images.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    
    unbatched = dataset.unbatch()
    images_labels = list(unbatched)
    
    plt.figure(figsize=(15, 5))
    
    for i, class_name in enumerate(class_names):
        count = 0
        for img, lbl in images_labels:
            if class_names[lbl.numpy()] == class_name:
                plt.subplot(1, len(class_names), i+1)
                plt.imshow(img.numpy())
                plt.title(class_name)
                plt.axis("off")
                count += 1
                if count >= samples_per_class:
                    break

    plt.tight_layout() # da se naslovi i slike ne preklapaju
    plt.savefig(save_path) 
    plt.close() 

