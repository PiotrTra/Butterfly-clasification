import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Ścieżka do folderu projektu
project_folder = 'C:\\Users\\poncz\\OneDrive\\Pulpit\\studiasrudia\\Inteligencja\\Projekt'

# Define number of classes (gatunki motyli)
train_csv = os.path.join(project_folder, 'Training_set.csv')
train_df = pd.read_csv(train_csv)  # Załaduj dane treningowe

num_classes = len(train_df['label'].unique())  # Ustal liczbę klas

# Define CNN model architecture
model = models.Sequential([
    Conv2D(32, (3, 3), 1, activation='relu', 
           input_shape=(224, 224, 3), kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    MaxPooling2D(),

    Conv2D(64, (3, 3), 1, activation='relu', 
           kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    MaxPooling2D(),

    Conv2D(128, (3, 3), 1, activation='relu', 
           kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    MaxPooling2D(),

    Flatten(),
    Dense(256, activation='relu', 
          kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))
])
# Compile the CNN model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Ścieżki do danych
train_data_dir = os.path.join(project_folder, 'train')
val_csv = os.path.join(project_folder, 'Validating_set.csv')
val_data_dir = os.path.join(project_folder, 'val')
test_csv = os.path.join(project_folder, 'Testing_set.csv')
test_data_dir = os.path.join(project_folder, 'test')

# Przygotowanie generatorów danych
train_datagen = ImageDataGenerator(
    rescale=1./255,                     # Skalowanie wartości pikseli
    rotation_range=30,                  # Losowe obracanie obrazów w zakresie 30 stopni
    width_shift_range=0.2,              # Losowe przesunięcia w poziomie (do 20% szerokości)
    height_shift_range=0.2,             # Losowe przesunięcia w pionie (do 20% wysokości)
    shear_range=0.2,                    # Losowe ścinanie obrazów
    zoom_range=0.2,                     # Losowe powiększanie/zmniejszanie obrazu
    horizontal_flip=True,               # Losowe odbijanie poziome
    fill_mode='nearest'                 # Wypełnianie brakujących pikseli najbliższą wartością
)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_data_dir,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=64,
    class_mode='sparse'
)

val_df = pd.read_csv(val_csv)
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=val_data_dir,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

test_df = pd.read_csv(test_csv)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_data_dir,
    x_col='filename',
    y_col=None,
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

# Sprawdź, czy istnieje zapisany model
model_path = os.path.join(project_folder, 'cnn_model_train_83_val_50.h5')
if os.path.exists(model_path):
    print("Wczytywanie zapisanego modelu...")
    model = load_model(model_path)
else:
    # Trenowanie modelu
    print("Training CNN Model...")
    cnn_history = model.fit(train_generator, validation_data=val_generator, epochs=50, verbose=1)

    # Zapisz model
    model.save(model_path)
    print(f"Model zapisany w {model_path}")

    # Zapisz wykresy
    def plot_training_history(history, title, save_path):
        plt.figure(figsize=(12, 4))
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{title} - Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{title} - Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Wykres zapisany w {save_path}")
        plt.show()

    plot_path = os.path.join(project_folder, 'training_history.png')
    plot_training_history(cnn_history, 'CNN Model', plot_path)

# Testowanie modelu na danych testowych
print("Testowanie modelu na danych testowych...")
cnn_predictions = model.predict(test_generator)

# Wyświetl przykładowe przewidywania
class_names = {v: k for k, v in train_generator.class_indices.items()}

def display_predictions(predictions, class_names, num_samples_per_graph=10):
    total_samples = len(predictions)
    num_graphs = (total_samples + num_samples_per_graph - 1) // num_samples_per_graph
    for graph_index in range(num_graphs):
        start_index = graph_index * num_samples_per_graph
        end_index = min(start_index + num_samples_per_graph, total_samples)
        plt.figure(figsize=(15, 10))
        for i, (img_path, pred) in enumerate(zip(
                test_generator.filenames[start_index:end_index],
                predictions[start_index:end_index])):
            plt.subplot(2, num_samples_per_graph // 2, i + 1)
            img = load_img(os.path.join(test_data_dir, img_path))
            plt.imshow(img)
            plt.title(f"{class_names[np.argmax(pred)]}\nConfidence: {np.max(pred):.2f}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

print("CNN Predictions:")
display_predictions(cnn_predictions, class_names)