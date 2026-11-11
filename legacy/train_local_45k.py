#Importovanje biblioteka
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import load_model
from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras import layers

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import SparseCategoricalCrossentropy  

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Definisanje podataka
main_path = 'Galaksije vece'
img_size = (150, 150)
batch_size = 64

#Load the data
Xtrain, Xval = image_dataset_from_directory(main_path, 
                                      subset='both', 
                                      validation_split=0.2,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      seed=123)

#Class names and count
classes = Xtrain.class_names
num_classes = len(classes)
print(f'Dataset classes: {classes}')

#Plotovanje 10 primjeraka
img, lab = next(iter(Xtrain))
plt.figure()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(img[i].numpy().astype('uint8'))
    plt.title(classes[lab[i]])
    plt.axis('off')
plt.show()

#Augmentacija
data_augmentation = Sequential(
  [
      layers.Input((img_size[0], img_size[1], 3)),
      layers.RandomFlip("horizontal"),
      layers.RandomRotation(0.25),
      layers.RandomZoom(0.1),
  ]
)

for i in range(10):
    aug_img = data_augmentation(img)
    plt.subplot(2, 5, i+1)
    plt.imshow(aug_img[0].numpy().astype('uint8'))
    plt.axis('off')
plt.show()

#Definisanje modela i parametara
def cnn_model(num_classes):
    model = Sequential([
        data_augmentation,
        
        # First convolutional block
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        # Second convolutional block
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        # Third convolutional block
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Fourth convolutional block
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Dropout for regularization
        layers.Dropout(0.3),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model

#Formiranje i ispis modela
model = cnn_model(num_classes)
#model = load_model('best_model.keras')
model.summary()

#Early stop and checkpoint callbacks
es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('best_model2.keras', save_best_only=True, monitor='val_accuracy', verbose=1)

#Training the model
history = model.fit(Xtrain,
                    epochs=30,
                    validation_data=Xval,
                    callbacks=[es, checkpoint],
                    verbose=1)

model.save("galaxy_cnn_model_DaveNet2.keras")

# Accuracy and loss plots
plt.figure()
plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.show()

#Konfuziona matrica i metrike
y_true = np.array([])
y_pred = np.array([])
for img, lab in Xval:
    y_true = np.concatenate([y_true, lab.numpy()])
    y_pred = np.concatenate([y_pred, np.argmax(model.predict(img, verbose=0), axis=1)])

print(f"Model accuracy: {100 * accuracy_score(y_true, y_pred):.2f}%")

cm = confusion_matrix(y_true, y_pred, normalize='true')
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot()
plt.show()
