import numpy as np
from keras.models import model_from_json, Sequential
import matplotlib.pyplot as plt
from keras.preprocessing import image

# Define emotion categories (ensure it matches the training order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the updated model architecture from JSON
with open('C:\\Users\\Hacker06\\Downloads\\Emotion_detection_with_CNN\\Emotion_detection_with_CNN-main\\model\\emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load model architecture with custom objects
emotion_model = model_from_json(loaded_model_json, custom_objects={'Sequential': Sequential})

# Load the updated model weights
emotion_model.load_weights('C:\\Users\\Hacker06\\Downloads\\Emotion_detection_with_CNN\\Emotion_detection_with_CNN-main\\model\\emotion_model_updated.h5')
print("Loaded updated model from disk")

# Compile the model
from tensorflow.keras.optimizers import Adam

emotion_model.compile(
    optimizer=Adam(learning_rate=0.0001),  # It adjusts the learning rate during training
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("Model compiled successfully")


# Function to preprocess image
def load_and_preprocess_image(img_path):
    """
    Load and preprocess the image for the model.
    """
    # Load the image in grayscale and resize to 48x48 (same size used during training)
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")

    # Convert the image to array format
    img_array = image.img_to_array(img)

    # Rescale pixel values (as the model was trained with rescaled data)
    img_array = img_array / 255.0

    # Expand dimensions to match the input shape expected by the model (1, 48, 48, 1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Function to predict emotion from image
def predict_emotion(img_path):
    """
    Predict the emotion for a given image.
    """
    # Preprocess the image
    processed_image = load_and_preprocess_image(img_path)

    # Make prediction using the trained model
    prediction = emotion_model.predict(processed_image)

    # Get the index of the highest confidence value
    emotion_index = np.argmax(prediction)

    # Get the corresponding emotion label
    predicted_emotion = emotion_dict[emotion_index]

    print(f"Predicted Emotion: {predicted_emotion}")

    # Display the image along with the predicted emotion
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted Emotion: {predicted_emotion}")
    plt.axis('off')
    plt.show()


# Test the updated model on a specific image (replace with your actual image file path)
test_image_path = 'C:\\Users\\Hacker06\\Downloads\\data\images\\validation\\angry\\966.jpg'  # Example path
predict_emotion(test_image_path)