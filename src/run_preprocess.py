from preprocess import load_train_val, load_test, count_images_per_class, show_sample_images
import matplotlib.pyplot as plt

train_ds, val_ds, class_names = load_train_val("data/train")
test_ds, _ = load_test("data/test")

class_counts = count_images_per_class(train_ds, class_names)

print("Broj slika po klasama u trening dataset-u:")
for cls, count in class_counts.items():
    print(f"- {cls}: {count}")


plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=45, ha='right')
plt.xlabel("Klasa")
plt.ylabel("Broj slika")
plt.tight_layout()
plt.savefig("samples/class_distribution.png")

show_sample_images(train_ds, class_names, samples_per_class=1, save_path="samples/train_samples.png")
