### EXP NO: 07

### .

# <p align = "center"> Convolutional Autoencoder for Image Denoising
</p>
  
## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Using autoencoder, we are trying to remove the noise added in the encoder part and tent to get the output which should be same as the input with minimal loss.
The dataset which is used is mnist dataset.

## Convolution Autoencoder Network Model
![OIP](https://user-images.githubusercontent.com/75235477/201523163-26427793-e99a-48de-a197-4b24db02b10c.jpg)

## DESIGN STEPS

### STEP 1:
Download and split the dataset into training and testing datasets.
### STEP 2:
Rescale the data as that the training is made easy.
### STEP 3:
Add noise factor.
### STEP 4:
Create a autoencoder model.
### STEP 5:
Compile and fit the created model.
### STEP 6:
Display the Original, Noisy and Reconstructed Image.

## PROGRAM
```
Developed by: Sri hari krishnan R
Reg No: 212219040150
```
```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))

x= layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
layers.MaxPooling2D((2,2),padding='same')(x)
x= layers.Conv2D(14, (3, 3), activation='relu', padding='same')(x)
layers.MaxPooling2D((2,2),padding='same')(x)
x= layers.Conv2D(7, (3, 3), activation='relu', padding='same')(x)

encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Encoder output dimension is ## Mention the dimention ##
x= layers.Conv2D(14, (3, 3), activation='relu', padding='same')(encoded)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)

decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))

decoded_imgs = autoencoder.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/75235334/201502294-bb0e34c3-90b2-4944-98fb-4556f1ac3569.png)

### Original vs Noisy Vs Reconstructed Image
![image](https://user-images.githubusercontent.com/75235334/201501482-214e7ab0-406a-4191-ac02-ae2fd4adbb48.png)

## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.
