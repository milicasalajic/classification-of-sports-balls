from preprocess import load_train_val, load_test
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# 1. učitaj podatke
train_ds, val_ds, class_names = load_train_val("data/train")
test_ds, _ = load_test("data/test") 

IMG_SIZE = (180, 180)

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc*100:.2f}%")

