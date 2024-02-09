import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Load and preprocess the images
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values
train_generator = train_datagen.flow_from_directory(
        'C:\\Users\\MST\\PycharmProjects\\RegNet-upload\\pretrain',  # Directory containing the training images
        target_size=(224, 224),  # Resize images to 224x224 (adjust as needed)
        batch_size=32,
        class_mode='binary')  # Adjust class_mode based on your dataset

# Step 2: Define your RegNet model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Step 3: Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the model on your dataset
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples // train_generator.batch_size,
      epochs=10)

# Step 5: Save the model for future use
model.save('regnet_model.h5')

# Optionally, you may evaluate the model on test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'C:\\Users\\MST\\PycharmProjects\\TF2-Pretrained-RegNets-master\\test_data',  # Directory containing the test images
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')
