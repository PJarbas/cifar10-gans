from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from model import GAN
from data_manager import DataManager

# usage: python train.py


class ModelTrain:
    """ Regular GAN that generates images using a random latent vector as input.
        While it works great we do not know the mapping of latent vector to the generated image.
    """

    def __init__(self):
        self.data_manager = DataManager()
        self.dataset = self.load_real_samples()
        self.generative_model = GAN()

    # load cifar training images
    def load_real_samples(self):
        """ Load cifar10 and scale from [0,255] to [-1,1]
            Generator uses tanh activation so rescale
            Original images to -1 to 1 to match the output of generator.

        Returns:
            x: _description_
        """

        train_x, _, _, _ = self.data_manager.cifar_10()

        # Convert to float and scale.
        x = train_x.astype('float32')
        x = (x - 127.5) / 127.5

        return x

    def generate_real_samples(self, n_samples):
        """ Pick a batch of random real samples to train the GAN
            In fact, we will train the GAN on a half batch of real images and another
            half batch of fake images. 
            For each real image we assign a label 1 and for fake we assign label 0.

        Args:
            n_samples (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        # choose random images
        ix = randint(0, self.dataset.shape[0], n_samples)

        # select the random images and assign it to X
        X = self.dataset[ix]

        # generate class labels and assign to y
        y = ones((n_samples, 1))  # Label=1 indicating they are real

        return X, y

    def generate_latent_points(self, latent_dim, n_samples):
        """ Generate n_samples number of latent vectors as input for the generator

        Args:
            latent_dim (_type_): _description_
            n_samples (_type_): _description_

        Returns:
            _type_: _description_
        """

        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        
        return x_input

    def generate_fake_samples(self, generator, latent_dim, n_samples):
        """ Use the generator to generate n fake examples, with class labels
            Supply the generator, latent_dim and number of samples as input.
            Use the above latent point generator to generate latent points. 

        Args:
            generator (_type_): _description_
            latent_dim (_type_): _description_
            n_samples (_type_): _description_

        Returns:
            _type_: _description_
        """

        # generate points in latent space
        x_input = self.generate_latent_points(latent_dim, n_samples)

        # predict using generator to generate fake samples.
        x = generator.predict(x_input)

        # Class labels will be 0 as these samples are fake.
        # Label=0 indicating they are fake
        y = zeros((n_samples, 1))

        return x, y

    def run(self, latent_dim=100, lr=0.0002, beta_1=0.5,
            n_nodes = 128 * 8 * 8, n_epochs=10, n_batch=128,
            loss='binary_crossentropy'):
        
        """ Train the generator and discriminator
            We loop through a number of epochs to train our Discriminator by first selecting
            A random batch of images from our true/real dataset.
            Then, generating a set of images using the generator.
            Feed both set of images into the Discriminator.
            Finally, set the loss parameters for both the real and fake images, as well as the combined loss.


        Args:
            latent_dim (int, optional): _description_. Defaults to 100.
            lr (float, optional): _description_. Defaults to 0.0002.
            beta_1 (float, optional): _description_. Defaults to 0.5.
            n_nodes (_type_, optional): _description_. Defaults to 128*8*8.
            n_epochs (int, optional): _description_. Defaults to 100.
            n_batch (int, optional): _description_. Defaults to 128.
            loss (str, optional): _description_. Defaults to 'binary_crossentropy'.
        """
        
        discriminator = self.generative_model.discriminator(lr=lr, beta_1=beta_1, loss=loss)
        
        generator = self.generative_model.generator(latent_dim=latent_dim, n_nodes =n_nodes)
        
        # create the gan
        gan_model = self.generative_model.build_gan(generator=generator, discriminator=discriminator,
                                                    lr=lr, beta_1=beta_1, loss=loss)
        
        bat_per_epo = int(self.dataset.shape[0] / n_batch)
        
        # the discriminator model is updated for a half batch of real samples
        half_batch = int(n_batch / 2)
        
        # and a half batch of fake samples, combined a single batch.
        # manually enumerate epochs and bacthes.
        for i in range(n_epochs):
            
            # enumerate batches over the training set
            for j in range(bat_per_epo):

                # Train the discriminator on real and fake images, separately (half batch each)
                # Research showed that separate training is more effective.
                # get randomly selected 'real' samples
                x_real, y_real = self.generate_real_samples(half_batch)
                
                # update discriminator model weights
                # train_on_batch allows you to update weights based on a collection
                # of samples you provide
                # Let us just capture loss and ignore accuracy value (2nd output below)
                d_loss_real, _ = discriminator.train_on_batch(x_real, y_real)

                # generate 'fake' examples
                x_fake, y_fake = self.generate_fake_samples(
                    generator, latent_dim, half_batch)
                
                # update discriminator model weights
                d_loss_fake, _ = discriminator.train_on_batch(x_fake, y_fake)

                # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #Average loss if you want to report single..

                # prepare points in latent space as input for the generator
                x_gan = self.generate_latent_points(latent_dim, n_batch)

                # The generator wants the discriminator to label the generated samples
                # as valid (ones)
                # This is where the generator is trying to trick discriminator into believing
                # the generated image is true (hence value of 1 for y)
                y_gan = ones((n_batch, 1))

                # Generator is part of combined model where it got directly linked with the discriminator
                # Train the generator with latent_dim as x and 1 as y.
                # Again, 1 as the output as it is adversarial and if generator did a great
                # job of folling the discriminator then the output would be 1 (true)
                # update the generator via the discriminator's error
                g_loss = gan_model.train_on_batch(x_gan, y_gan)

                # Print losses on this batch
                print('Epoch>%d, Batch %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                      (i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))
                
        # save the generator model
        generator.save(f"cifar_generator_{n_epochs}epochs.h5")


if __name__ == "__main__":
    model_train = ModelTrain()
    model_train.run(n_epochs=100)
