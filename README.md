# Klasifikacija Sportskih Loptica

Projekat za klasifikaciju sportskih loptiča korišćenjem različitih deep learning modela. Implementirani su Simple CNN, Deep CNN, RNN i Transfer Learning (MobileNetV2) modeli za klasifikaciju 7 različitih tipova sportskih loptiča.

## 📋 Sadržaj

- [Opis projekta](#opis-projekta)
- [Struktura projekta](#struktura-projekta)
- [Instalacija](#instalacija)
- [Pokretanje](#pokretanje)
- [Modeli](#modeli)
- [Rezultati](#rezultati)
- [Fajlovi](#fajlovi)

## 🎯 Opis projekta

Ovaj projekat implementira sistem za automatsku klasifikaciju sportskih loptiča koristeći različite arhitekture neuronskih mreža. Dataset sadrži slike 7 različitih tipova sportskih loptiča:

- **Basketball** (340 trening, 100 test slika)
- **Billiard Ball** (646 trening, 162 test slika)  
- **Football** (604 trening, 151 test slika)
- **Hockey Puck** (390 trening, 98 test slika)
- **Rugby Ball** (493 trening, 124 test slika)
- **Tennis Ball** (490 trening, 123 test slika)
- **Volleyball** (432 trening, 109 test slika)

## 📁 Struktura projekta

```
classification-of-sports-balls/
├── data/                          # Dataset
│   ├── train/                     # Trening podaci (po klasama)
│   └── test/                      # Test podaci (po klasama)
├── src/                          # Izvorni kod
│   ├── preprocess.py             # Preprocessing i učitavanje podataka
│   ├── simple_cnn_train.py       # Simple CNN model
│   ├── deep_cnn_train.py         # Deep CNN model
│   ├── rnn_train.py              # RNN model
│   ├── train_transfer.py         # Transfer Learning model
│   └── run_preprocess.py         # Pokretanje preprocessing-a
├── metrics/                       # Evaluacija modela
│   ├── models_fit.py             # Treniranje svih modela
│   ├── evaluate_models.py        # Evaluacija i poređenje modela
│   └── utils.py                  # Pomoćne funkcije
├── models/                        # Sačuvani modeli (.h5 fajlovi)
├── plots/                         # Generisani grafikoni i matrice
├── samples/                       # Uzorci slika i distribucija klasa
└── README.md                      # Dokumentacija
```

## 🚀 Instalacija

### Potrebne biblioteke

```bash
pip install tensorflow
pip install matplotlib
pip install numpy
pip install scikit-learn
```

### Kreiranje potrebnih direktorijuma

```bash
mkdir models plots samples
```

## ▶️ Pokretanje

### 1. Preprocessing podataka

```bash
cd src
python run_preprocess.py
```

Ovaj korak će:
- Učitati trening i test podatke
- Prikazati distribuciju klasa
- Generisati uzorke slika u `samples/` direktorijumu

### 2. Treniranje modela

```bash
cd metrics
python models_fit.py
```

Ovaj korak će trenirati sve 4 modela i sačuvati ih u `models/` direktorijumu:
- `simple_cnn.h5`
- `deep_cnn.h5` 
- `rnn_model.h5`
- `transfer_model.h5`

### 3. Evaluacija modela

```bash
cd metrics
python evaluate_models.py
```

Ovaj korak će:
- Evaluirati sve modele na test skupu
- Generisati confusion matrice za svaki model
- Prikazati uzorke test slika sa predikcijama
- Kreirati uporedne grafikone accuracy i loss-a

## 🤖 Modeli

### 1. Simple CNN
- **Arhitektura**: 1 Conv2D sloj + MaxPooling + Dense slojevi
- **Parametri**: 32 filtera, 3x3 kernel
- **Slojevi**: Conv2D → MaxPooling → Flatten → Dense(64) → Dropout → Dense(7)

### 2. Deep CNN  
- **Arhitektura**: 3 Conv2D sloja + MaxPooling + Dense slojevi
- **Parametri**: 32, 64, 128 filtera, 3x3 kernel
- **Slojevi**: Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → Conv2D(128) → MaxPooling → Flatten → Dense(128) → Dropout → Dense(7)

### 3. RNN
- **Arhitektura**: Reshape + SimpleRNN + Dense slojevi
- **Parametri**: 128 RNN jedinica
- **Slojevi**: Reshape → SimpleRNN(128) → Dense(64) → Dropout → Dense(7)

### 4. Transfer Learning (MobileNetV2)
- **Bazni model**: MobileNetV2 (pre-trained na ImageNet)
- **Arhitektura**: Zamrznuta baza + custom head
- **Slojevi**: MobileNetV2 (frozen) → GlobalAveragePooling2D → Dense(128) → Dropout → Dense(7)

## 📊 Rezultati

Rezultati se čuvaju u `plots/` direktorijumu:

- **Confusion matrice**: `{model_name}_confusion_matrix.png`
- **Test slike sa predikcijama**: `{model_name}_test_images.png`
- **Uporedni accuracy**: `accuracy_comparison.png`
- **Uporedni loss**: `loss_comparison.png`

## 📄 Fajlovi

### `src/preprocess.py`
- `load_train_val()`: Učitava trening i validacione podatke
- `load_test()`: Učitava test podatke
- `count_images_per_class()`: Broji slike po klasama
- `show_sample_images()`: Prikazuje uzorke slika

### `src/simple_cnn_train.py`
- `build_simple_cnn()`: Kreira Simple CNN model

### `src/deep_cnn_train.py`
- `build_deep_cnn()`: Kreira Deep CNN model

### `src/rnn_train.py`
- `build_rnn()`: Kreira RNN model

### `src/train_transfer.py`
- `build_transfer_cnn()`: Kreira Transfer Learning model

### `metrics/models_fit.py`
- Trenira sve 4 modela sekvencijalno
- Koristi Early Stopping callback
- Čuva modele u `.h5` formatu

### `metrics/evaluate_models.py`
- Evaluira sve modele na test skupu
- Generiše confusion matrice
- Prikazuje uzorke test slika
- Kreira uporedne grafikone

### `metrics/utils.py`
- Pomoćne funkcije za plotovanje
- `plot_training_history()`: Prikazuje training history
- `plot_confusion_matrix()`: Generiše confusion matrice

## 🔧 Konfiguracija

### Osnovni parametri (u `preprocess.py`):
- `IMG_SIZE = (180, 180)`: Dimenzije slika
- `BATCH_SIZE = 32`: Veličina batch-a
- `VAL_SPLIT = 0.2`: Procenat podataka za validaciju

### Data augmentation:
- Horizontal flip
- Random rotation (do 10%)
- Random zoom (do 10%)

## 📈 Performance

Modeli se treniraju sa:
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Early Stopping**: Patience = 3, monitor = val_loss
- **Epochs**: 10-15 (zavisi od modela)

## 🤝 Doprinos

Za doprinos projektu:
1. Fork-ujte repository
2. Kreirajte feature branch
3. Commit-ujte promene
4. Push-ujte na branch
5. Otvorite Pull Request

## 📝 Licenca

Ovaj projekat je kreiran u edukativne svrhe.
