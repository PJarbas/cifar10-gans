import tensorflow as tf
from matplotlib import pyplot as plt


class DataManager:
    """_summary_
    """

    def __init__(self):
        pass

    def cifar_10(self):
        """
        Ten classes dataset: 'airplane', 'automobile', 'bird', 'cat', 'deer',
                             'dog', 'frog', 'horse', 'ship', 'truck'
        Returns:
            ndarray: training_images, training_labels, validation_images, validation_labels
        """

        (training_images, training_labels), (validation_images,
                                             validation_labels) = tf.keras.datasets.cifar10.load_data()

        return training_images, training_labels, validation_images, validation_labels

    def show_images(self, images_number=25):
        """plot images
        """

        training_images, _, _, _ = self.cifar_10()

        for i in range(images_number):
            plt.subplot(5, 5, 1 + i)
            plt.axis('off')
            plt.imshow(training_images[i])
        plt.show()
