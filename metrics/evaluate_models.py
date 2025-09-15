import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.models import load_model
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import load_train_val, load_test
from utils import plot_confusion_matrix

# -------------------------
# 1. Učitaj podatke
# -------------------------
train_ds, val_ds, class_names = load_train_val("data/train")
test_ds, test_labels = load_test("data/test")

# -------------------------
# 2. Lista svih modela i imena
# -------------------------
models_info = {
    "Simple_CNN": "models/simple_cnn.h5",
    "Deep_CNN": "models/deep_cnn.h5",
    "RNN": "models/rnn_model.h5",
    "Transfer_CNN_MobileNetV2": "models/transfer_model.h5"
}

# -------------------------
# 3. Pomoćne funkcije
# -------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, model_name="model"):
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig(f"plots/{model_name}_confusion_matrix.png")
    plt.close()


def show_test_images_random(model, test_ds, class_names, model_name="model"):
    """
    Uzima po jednu random sliku iz svake klase i prikazuje predikcije modela.
    """
    # Sacuvaj sve slike po klasi
    class_to_images = {i: [] for i in range(len(class_names))}

    for images, labels in test_ds:
        for img, lbl in zip(images, labels):
            lbl_int = int(lbl.numpy())
            class_to_images[lbl_int].append(img)
    
    # Izaberi po jednu random sliku iz svake klase
    images_list = []
    labels_list = []
    for cls, imgs in class_to_images.items():
        if imgs:
            chosen_img = random.choice(imgs)
            images_list.append(chosen_img)
            labels_list.append(cls)
    
    # Napravi predikcije
    images_stack = tf.stack(images_list)
    preds = model.predict(images_stack)
    pred_labels = np.argmax(preds, axis=1)

    # Prikaz
    plt.figure(figsize=(12, 12))
    for i, (img, true_lbl, pred_lbl) in enumerate(zip(images_list, labels_list, pred_labels)):
        ax = plt.subplot(3, 3, i + 1)
        img_display = img.numpy()
        if img_display.max() <= 1.0:
            img_display = (img_display * 255).astype("uint8")
        else:
            img_display = img_display.astype("uint8")

        plt.imshow(img_display)
        plt.title(f"T: {class_names[true_lbl]}, P: {class_names[pred_lbl]}")
        plt.axis("off")

    plt.tight_layout()
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig(f"plots/{model_name}_test_images.png")
    plt.close()



# -------------------------
# 4. Evaluacija svih modela
# -------------------------
for model_name, model_path in models_info.items():
    print(f"\n--- Evaluacija modela: {model_name} ---")
    model = load_model(model_path)

    # Test accuracy i loss
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")

    # Konfuziona matrica
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    plot_confusion_matrix(y_true, y_pred, class_names, model_name=model_name)

    # Prikaz nekoliko test slika sa predikcijama
    show_test_images_random(model, test_ds, class_names, model_name=model_name)



# -------------------------
# 5. Uporedni bar plot za test accuracy
# -------------------------
model_names = []
accuracies = []

for model_name, model_path in models_info.items():
    model = load_model(model_path)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    model_names.append(model_name)
    accuracies.append(test_acc*100)  # pretvori u procente

plt.figure(figsize=(10,6))
bars = plt.bar(model_names, accuracies, color=['skyblue','lightgreen','salmon','orange'])
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)
plt.title("Uporedni test accuracy svih modela")

# Dodaj vrednosti iznad bara
for bar, acc in zip(bars, accuracies):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{acc:.2f}%", ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()

if not os.path.exists("plots"):
    os.makedirs("plots")
plt.savefig("plots/accuracy_comparison.png")
plt.close()


# -------------------------
# 6. Uporedni bar plot za test loss
# -------------------------
model_names = []
losses = []

for model_name, model_path in models_info.items():
    model = load_model(model_path)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    model_names.append(model_name)
    losses.append(test_loss)  # čuvamo loss

plt.figure(figsize=(10,6))
bars = plt.bar(model_names, losses, color=['skyblue','lightgreen','salmon','orange'])
plt.ylabel("Test Loss")
plt.title("Uporedni test loss svih modela")

# Dodaj vrednosti iznad bara
for bar, loss in zip(bars, losses):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{loss:.4f}", ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()

if not os.path.exists("plots"):
    os.makedirs("plots")
plt.savefig("plots/loss_comparison.png")
plt.close()

