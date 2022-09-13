from keras.models import load_model
from matplotlib import pyplot as plt
from train import ModelTrain
import numpy as np

# Generate cifar10 dataset with GANS
# Adapted from: https://machinelearningmastery.com/
# 			  @digitalSreeni


def show_images(examples, n):
    for i in range(n*n):
        plt.subplot(n, n, 1+i)
        plt.axis("off")
        plt.imshow(examples[i, :, :, :])
    plt.show()

# load model
# Model trained for 10 epochs
MODEL_FILE_NAME = 'cifar_generator_100epochs.h5'
model = load_model(MODEL_FILE_NAME)

# generate images
# Latent dim and n_samples
latent_points = ModelTrain().generate_latent_points(100, 25)

# generate images
data = model.predict(latent_points)

# scale from [-1,1] to [0,1]
data = (data + 1) / 2.0

data = (data*255).astype(np.uint8)

# plot the result
show_images(data, 5)
