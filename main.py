import os
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from skimage.feature import hog
from skimage import io, filters, transform
from sklearn.metrics.pairwise import cosine_similarity

IMAGE_SIZE = (128, 128)

data_directory = "./dataset/train_data"  

def load_database():
    database = []
    labels = []
    files = [f for f in os.listdir(data_directory) if f.endswith('.bmp')]

    for file_name in files:
        base_id = file_name.split("_")[0]
        image_path = os.path.join(data_directory, file_name)
        image = io.imread(image_path, as_gray=True)
        image = transform.resize(image, IMAGE_SIZE, anti_aliasing=True)  
        image = filters.gaussian(image, sigma=1)
        fd, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        database.append(fd)
        labels.append(base_id)

    return np.array(database), np.array(labels)

def search_fingerprint():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp;*.jpg;*.png;*.tif")])

    if not file_path:
        status_message.set("Nijedna slika nije odabrana.")
        return

    query_image = io.imread(file_path, as_gray=True)
    query_image = transform.resize(query_image, IMAGE_SIZE, anti_aliasing=True)  
    query_image = filters.gaussian(query_image, sigma=1)
    query_features, _ = hog(query_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    similarities = cosine_similarity([query_features], database_features)
    best_match_idx = np.argmax(similarities)
    best_match_label = database_labels[best_match_idx]
    similarity_score = similarities[0, best_match_idx]

    status_message.set(f"Najbolji pogodak: {best_match_label} s sličnošću od {similarity_score * 100:.2f}%")

ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Fingerprint Search")
root.geometry("500x300")

status_message = ctk.StringVar()
status_message.set("Učitavam bazu otisaka...")

database_features, database_labels = load_database()
status_message.set("Baza uspješno učitana.")

ctk.CTkLabel(root, text="Odaberite otisak prsta za pretraživanje:", font=("Arial", 16)).pack(pady=20)
ctk.CTkButton(root, text="Odaberi sliku", command=search_fingerprint, width=200, height=40).pack(pady=10)
ctk.CTkLabel(root, textvariable=status_message, font=("Arial", 14), wraplength=400, justify="center").pack(pady=20)

root.mainloop()
