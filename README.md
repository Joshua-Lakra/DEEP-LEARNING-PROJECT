# DEEP-LEARNING-PROJECT

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : JOSHUA ATULYA LAKRA

*INTERN ID* : CT08DG1187

*DOMAIN* : DATA SCIENCE

*DURATION* : 8 WEEKS

*MENTOR* : Neela Santhosh

*Description* :

Deep learning has revolutionized the field of artificial intelligence, especially in tasks involving complex data such as images and natural language. In Task 2, the objective is to implement a deep learning model for image classification using TensorFlow, a leading open-source library developed by Google for machine learning and neural networks.

The dataset used for this task is MNIST, which consists of 70,000 images of handwritten digits (0 through 9). Each image is 28x28 pixels in grayscale. This dataset is considered a “Hello World” for deep learning and provides a great way to understand the basics of Convolutional Neural Networks (CNNs).

The first step is to load and preprocess the data. The dataset is available directly via tf.keras.datasets.mnist. The images are normalized by dividing the pixel values by 255.0 to scale them between 0 and 1, which improves model convergence. Since TensorFlow expects image data with a channel dimension, we reshape the input to (28, 28, 1).

Next, a CNN model is built using TensorFlow’s Sequential API. CNNs are particularly effective for image classification because they preserve the spatial relationships in data through convolutional and pooling layers. Our architecture includes:

A Conv2D layer with 32 filters and a 3x3 kernel, followed by a MaxPooling2D layer to downsample the feature maps.

A second Conv2D-MaxPooling2D pair with 64 filters to extract deeper features.

A Flatten layer to convert the 2D features into a 1D vector.

A Dense layer with 64 neurons and ReLU activation for learning complex patterns.

A final Dense layer with 10 neurons (for 10 digits) and softmax activation to output class probabilities.

The model is compiled using:

optimizer='adam' which is efficient and adaptive for most deep learning tasks.

loss='sparse_categorical_crossentropy' because we are dealing with multi-class classification.

metrics=['accuracy'] to track performance.

The model is then trained for 5 epochs using the .fit() method, with validation data provided to monitor overfitting.

After training, the accuracy and loss over epochs are visualized using Matplotlib. This helps understand how the model learned during each epoch. Plotting history.history['accuracy'] and ['val_accuracy'] gives insight into model performance on both training and unseen data.

We also generate predictions for the first few test images and visualize them along with the predicted labels. This part demonstrates real-world application, showing how the model interprets input and gives output.

Overall, Task 2 is a practical deep learning exercise covering the full workflow: data preparation, model design, training, evaluation, and prediction. It introduces important concepts like convolutional layers, overfitting prevention through validation, and result interpretation. This foundational knowledge prepares you for more complex problems involving image classification or natural language processing.


*Output* :

<img width="1005" height="585" alt="Image" src="https://github.com/user-attachments/assets/c67e1813-2c7a-42ad-a9ae-7270fc8ba318" />
<img width="994" height="582" alt="Image" src="https://github.com/user-attachments/assets/bf642871-40ce-4317-b058-b486f0c9f088" />
<img width="1249" height="584" alt="Image" src="https://github.com/user-attachments/assets/6ce3d1c2-1721-44bb-bf51-83578096266f" />
