import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import load_train_val, load_test
from src.simple_cnn_train import build_simple_cnn
from src.deep_cnn_train import build_deep_cnn
from src.rnn_train import build_rnn
from src.train_transfer import build_transfer_cnn  
from metrics.utils import plot_training_history, plot_confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np

# 1. Učitaj podatke
train_ds, val_ds, class_names = load_train_val("data/train")
test_ds, test_labels = load_test("data/test")

IMG_SIZE = (180, 180, 3)
num_classes = len(class_names)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# -------------------------
# 1. Simple CNN
# -------------------------
simple_cnn = build_simple_cnn(IMG_SIZE, num_classes)
history_simple = simple_cnn.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stop])
simple_cnn.save("models/simple_cnn.h5")  # čuvanje modela

# -------------------------
# 2. Deep CNN
# -------------------------
deep_cnn = build_deep_cnn(IMG_SIZE, num_classes)
history_deep = deep_cnn.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stop])
deep_cnn.save("models/deep_cnn.h5")      # čuvanje modela

# -------------------------
# 3. RNN
# -------------------------
rnn_model = build_rnn(IMG_SIZE, num_classes)
history_rnn = rnn_model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[early_stop])
rnn_model.save("models/rnn_model.h5")    # čuvanje modela

# -------------------------
# 4. Transfer Learning (MobileNetV2)
# -------------------------
transfer_model = build_transfer_cnn(IMG_SIZE, num_classes)
history_transfer = transfer_model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[early_stop])
transfer_model.save("models/transfer_model.h5")  # čuvanje modela
