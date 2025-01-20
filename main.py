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
data_directory = "./dataset/train_data"  # Ovdje zamijeniti stvarnom putanjom

# Učitavanje svih .tif datoteka u direktoriju
all_files = [f for f in os.listdir(data_directory) if f.endswith('.bmp')]

# Grupiranje podataka po ID-ovima (npr. 101, 102, ..., 110)
data = []
labels = []

for file_name in all_files:
    base_id = file_name.split("_")[0]  # Ekstrahiranje ID-a (npr. "101")
    
    # Učitavanje slike
    image_path = os.path.join(data_directory, file_name)
    image = io.imread(image_path, as_gray=True)  # Učitavanje slike u grayscale
    
    # Predprocesiranje - uklanjanje šuma (Gaussovo zamućenje)
    image = filters.gaussian(image, sigma=1)

    # Ekstrakcija HOG značajki
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    
    # Dodavanje značajki i oznake u liste
    data.append(fd)
    labels.append(base_id)  # ID otiska kao labela

# Provjera ravnoteže podataka prije podjele na trening i test skupove
print("Distribucija oznaka prije podjele:", Counter(labels))

# Pretvaranje podataka u numpy array
X = np.array(data)
y = np.array(labels)

# Podjela podataka na trening i test skupove (80% za trening, 20% za testiranje)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
# Ispis broja slika u direktoriju


# Vizualizacija HOG značajki za prvi uzorak
image_path = os.path.join(data_directory, all_files[random.randint(0,len(all_files)-1)])  
image = io.imread(image_path, as_gray=True)
fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original Image')
plt.subplot(122)
plt.imshow(hog_image, cmap=plt.cm.gray)
plt.title('HOG Features')
plt.show()