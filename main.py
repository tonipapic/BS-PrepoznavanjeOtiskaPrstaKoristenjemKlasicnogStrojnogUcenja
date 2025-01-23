import os
import random
import numpy as np
from skimage.feature import hog
from sklearn import svm
from skimage import io, filters
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt

data_directory = "./dataset/train_data"

all_files = [f for f in os.listdir(data_directory) if f.endswith('.bmp')]

labels = []
grouped_data = {}

for file_name in all_files:
    base_id = file_name.split("_")[0]  
    
    # Učitavanje slike
    image_path = os.path.join(data_directory, file_name)
    image = io.imread(image_path, as_gray=True)  
    
    image = filters.gaussian(image, sigma=1)
    
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    
    if base_id not in grouped_data:
        grouped_data[base_id] = []
    grouped_data[base_id].append(fd)
    labels.append(base_id)

print("Distribucija oznaka prije podjele:", Counter(labels))

data_train = []
data_test = []
labels_train = []
labels_test = []

for base_id, features in grouped_data.items():
    random.shuffle(features)
    
    split_index = int(len(features) * 0.8)
    
    data_train.extend(features[:split_index])
    data_test.extend(features[split_index:])
    labels_train.extend([base_id] * len(features[:split_index]))
    labels_test.extend([base_id] * len(features[split_index:]))

print("Distribucija oznaka u trening skupu:", Counter(labels_train))
print("Distribucija oznaka u test skupu:", Counter(labels_test))

X_train = np.array(data_train)
X_test = np.array(data_test)
y_train = np.array(labels_train)
y_test = np.array(labels_test)

model = svm.SVC(kernel='linear')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Točnost modela: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
tp = cm[0, 0]  
fn = cm[0, 1]  
fp = cm[1, 0]  
tn = cm[1, 1]  

FAR = fp / (fp + tn) if (fp + tn) != 0 else 0

FRR = fn / (fn + tp) if (fn + tp) != 0 else 0

print(f"FAR (False Acceptance Rate): {FAR * 100:.2f}%")
print(f"FRR (False Rejection Rate): {FRR * 100:.2f}%")

image_path = os.path.join(data_directory, all_files[random.randint(0,len(data_directory)-1)]) 
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
