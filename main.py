import os
import random
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from skimage import io, filters
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

# Putanja do direktorija sa slikama
data_directory = "./dataset/train_data"

# Učitavanje svih .bmp datoteka u direktoriju
all_files = [f for f in os.listdir(data_directory) if f.endswith('.bmp')]

# Grupiranje podataka po ID-ovima (npr. 101, 102, ..., 110)
data = []
labels = []
grouped_data = {}

# Grupiraj slike prema ID-ovima
for file_name in all_files:
    base_id = file_name.split("_")[0]  # Ekstrahiranje ID-a (npr. "101")
    
    # Učitavanje slike
    image_path = os.path.join(data_directory, file_name)
    image = io.imread(image_path, as_gray=True)  # Učitavanje slike u grayscale
    
    # Predprocesiranje - uklanjanje šuma (Gaussovo zamućenje)
    image = filters.gaussian(image, sigma=1)
    
    # Ekstrakcija HOG značajki
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    
    # Dodavanje slike u grupu prema ID-u
    if base_id not in grouped_data:
        grouped_data[base_id] = []
    grouped_data[base_id].append(fd)
    labels.append(base_id)

# Provjera ravnoteže podataka prije podjele na trening i test skupove
print("Distribucija oznaka prije podjele:", Counter(labels))

# Podjela podataka unutar svake klase (npr. 2 slike iz svake klase za test, ostatak za trening)
data_train = []
data_test = []
labels_train = []
labels_test = []

for base_id, features in grouped_data.items():
    # Nasumično pomiješaj slike u ovoj klasi
    random.shuffle(features)
    
    # Odaberi 80% za trening i 20% za test
    split_index = int(len(features) * 0.8)
    
    # Dodaj odabrane podatke u odgovarajuće skupove
    data_train.extend(features[:split_index])
    data_test.extend(features[split_index:])
    labels_train.extend([base_id] * len(features[:split_index]))
    labels_test.extend([base_id] * len(features[split_index:]))

# Provjera ravnoteže podataka nakon podjele
print("Distribucija oznaka u trening skupu:", Counter(labels_train))
print("Distribucija oznaka u test skupu:", Counter(labels_test))

# Pretvaranje podataka u numpy array
X_train = np.array(data_train)
X_test = np.array(data_test)
y_train = np.array(labels_train)
y_test = np.array(labels_test)

# Kreiranje SVM modela
model = svm.SVC(kernel='linear')

# Trening modela
model.fit(X_train, y_train)

# Predviđanje na testnim podacima
y_pred = model.predict(X_test)

# Evaluacija modela - točnost
accuracy = accuracy_score(y_test, y_pred)
print(f"Točnost modela: {accuracy * 100:.2f}%")

# Evaluacija modela - FAR (False Acceptance Rate) i FRR (False Rejection Rate)
cm = confusion_matrix(y_test, y_pred)
tp = cm[0, 0]  # True Positives
fn = cm[0, 1]  # False Negatives
fp = cm[1, 0]  # False Positives
tn = cm[1, 1]  # True Negatives

# FAR (False Acceptance Rate)
FAR = fp / (fp + tn) if (fp + tn) != 0 else 0

# FRR (False Rejection Rate)
FRR = fn / (fn + tp) if (fn + tp) != 0 else 0

print(f"FAR (False Acceptance Rate): {FAR * 100:.2f}%")
print(f"FRR (False Rejection Rate): {FRR * 100:.2f}%")
