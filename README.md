Animal Classifier

This is a program that can classify images of animals into 10 different categories using a convolutional neural network.

Requirements
Python 3.6 or later
OpenCV
NumPy
TensorFlow
scikit-learn

Usage: 
-Download the repository and navigate to the project directory.

-Place your images in the "raw-img" folder in the following format:

raw-img/
├── 1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
...

or use dataset, provided by Corrado Alessio (https://www.kaggle.com/datasets/alessiocorrado99/animals10). Keep in mind to change folder names to numbers.

Run AI.py and it will train your model and save it. With provided dataset, it might take some time to train it (couple of hours).

Model will be saved as .h5 file. To test model, use model test and new image of animal.



