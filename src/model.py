from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout


class GAN:
    """ Define generator and discriminator.
    """
    def __init__(self):
        pass

    def discriminator(self, lr, beta_1, loss, in_shape=(32,32,3)):
        """Define the standalone discriminator model
           Given an input image, the Discriminator outputs the likelihood of the image being real.
           Binary classification - true or false (1 or 0). So using sigmoid activation.

        Args:
            learning rate:
            beta_1:
            in_shape (tuple, optional): _description_. Defaults to (32,32,3).

        Returns:
            model: _description_
        """
        model = Sequential()
     
        model.add(Conv2D(128, (3,3), strides=(2,2),
                         padding='same', input_shape=in_shape))
       
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))      
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        
        # compile model
        opt = Adam(lr=lr, beta_1=beta_1)
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
        return model

    def generator(self, latent_dim, n_nodes):
        """ Define the standalone generator model
            Given input of latent vector, the Generator produces an image.(here: 32x32)
            latent_dim, for example, can be 100, 1D array of size 100.

        Args:
            latent_dim (_type_): the dimension of the latent vector (e.g., 100)
            n_nodes (_type_): _description_

        Returns:
            model: _description_
        """
        
        model = Sequential() 
        model.add(Dense(n_nodes, input_dim=latent_dim)) 
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((8, 8, 128)))
         
        # upsample to 16x16
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        
        # upsample to 32x32
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        
        # generate
        model.add(Conv2D(3, (8,8), activation='tanh', padding='same'))
        return model

    def build_gan(self, generator, discriminator, lr=0.01, beta_1=0.5, loss='binary_crossentropy'):
        
        """ Define the combined generator and discriminator model, for updating the generator
            Discriminator is trained separately so here only generator will be trained by keeping
            the discriminator constant.
        """
        
        # Discriminator is trained separately. So set to not trainable.
        discriminator.trainable = False
        
        # connect generator and discriminator
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        
        # compile model
        opt = Adam(lr=lr, beta_1=beta_1)
        model.compile(loss=loss, optimizer=opt)
        
        return model
        
