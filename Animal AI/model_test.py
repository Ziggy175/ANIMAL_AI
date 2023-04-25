import tensorflow as tf
import numpy as np
import cv2

path_test = r'testne-slike\slika1.jpg'
IMAGE_SIZE = (128, 128)
model = tf.keras.models.load_model('animal_classifier.h5')

# Load a test image and resize it to the desired size
img = cv2.imread(path_test, 0)
img = cv2.resize(path_test, IMAGE_SIZE)
cv2.imwrite(path_test,img)
img = np.reshape(img, (-1, 128, 128, 1))

IMAGE_SIZE = (128, 128)




# Normalize the image data
img = img / 255.0

# Make a prediction on the test image
prediction = model.predict(img)

prediction_class = np.argmax(prediction, axis=1)
if prediction_class == 1:
    print("Butterfly")
elif prediction_class == 2:
    print("Cat")
elif prediction_class == 3:
    print("Chicken")
elif prediction_class == 4:
    print("Cow")
elif prediction_class == 5:
    print("Dog")
elif prediction_class == 6:
    print("Elephant")
elif prediction_class == 7:
    print("Horse")
elif prediction_class == 8:
    print("Sheep")
elif prediction_class == 9:
    print("Spider")
elif prediction_class == 10:
    print("Squirrel")
else:
    print("Error")

# Print the predicted class probabilities
