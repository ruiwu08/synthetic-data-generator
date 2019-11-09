print("Let's begin!")

from keras.layers import Input, Dense, Flatten, Dropout, Reshape
from keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import image
from keras.utils.generic_utils import Progbar
from keras.initializers import RandomNormal

from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import keras.backend as K

import matplotlib.pyplot as plt

import sys
import numpy as np
import pandas as pd

#NB_EPOCHS = 20
#D_ITERS = 5
#BATCH_SIZE = 128
#BASE_N_COUNT = 128
#LATENT_SIZE = 100
#SAMPLE_SIZE = 8192

#THIS PROCESS CSV FUNCTION IS ONLY USED WHEN RUNNING ENGINGE.PY ALONE AND IS NOT REFERENCED IN COLUMNS.PY
def process_csv(csv_path):
    # Input: The path location of the CSV
    # Outputs: 
    #       1. The CSV scaled down to be between -1 and 1
    #       2. An array of maximum absolute values for each column. 

    real_full_data = pd.read_csv(csv_path, header=0)
    real_full_data = real_full_data.dropna()

    # Store the maximum absolute value of each column.
    col_max_array = real_full_data.abs().max().to_frame()
    
    # Scale the data to be between -1, 1
    real_scaled_data = real_full_data / real_full_data.abs().max()

    return real_scaled_data, col_max_array, real_full_data

class GAN():
    def __init__(self, real_scaled_data, col_max_array):
        self.real_data = real_scaled_data
        self.col_max_array = pd.DataFrame(col_max_array).T
        self.data_dim = len(real_scaled_data.columns)
        self.latent_dim = 100
        self.base_n_count = 256
        self.d_iters = 5

        optimizer = RMSprop(lr=0.00005)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss = wasserstein_loss,
            optimizer = optimizer,
        )

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim, ), name = 'input_z')
        generated_data = self.generator(z)

        is_fake = self.discriminator(generated_data)

        self.combined = Model(z, is_fake)
        self.combined.get_layer('D').trainable = False
        self.combined.compile(loss = wasserstein_loss, optimizer = optimizer)

    def build_generator(self):

        weight_init = RandomNormal(mean = 0., stddev = 0.02)
        model = Sequential()
        
        model.add(Dense(self.base_n_count, input_dim = self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.base_n_count*2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.base_n_count*4))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(self.data_dim, activation = 'tanh'))

        noise = Input(shape=(self.latent_dim,))
        fake_data = model(noise)

        print(model.summary())

        return Model(noise, fake_data, name = 'G')

    def build_discriminator(self):

        weight_init = RandomNormal(mean = 0., stddev = 0.02)
        model = Sequential()

        model.add(Dense(self.base_n_count * 4, input_dim = self.data_dim))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.3))
        model.add(Dense(self.base_n_count * 2))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.3))
        model.add(Dense(self.base_n_count))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation = 'linear'))

        print(model.summary())

        data_features = Input(shape = (self.data_dim, ))
        is_fake = model(data_features)

        return Model(data_features, is_fake, name = 'D')

    def train(self, epochs = 30, batch_size = 128, sample_size = 8192):
        self.fake_data_list = []
        self.epoch_gen_loss = []
        self.epoch_disc_true_loss = []
        self.epoch_disc_fake_loss = []
        nb = int(sample_size / batch_size) * epochs
        rounds = int(sample_size / batch_size)

        progress_bar = Progbar(target = nb)

        for index in range(nb):
            x_train = self.real_data.sample(sample_size)
            progress_bar.update(index)
            for d_it in range(self.d_iters):
                # unfreeze D
                self.discriminator.trainable = True
                for l in self.discriminator.layers:
                    l.trainable = True

                # clip D weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -0.01, 0.01) for w in weights]
                    l.set_weights(weights)

                # Maximize D output on reals == minimize -1*(D(real)) and get a batch of real data
                data_index = np.random.choice(len(x_train), batch_size, replace = False)
                data_batch = x_train.values[data_index]

                self.epoch_disc_true_loss.append(self.discriminator.train_on_batch(data_batch, -np.ones(batch_size)))

                # Minimize D output on fakes
                # generate a new batch of noise
                noise = np.random.normal(loc=0.0, scale=1, size=(int(batch_size), self.latent_dim))

                generated_data = self.generator.predict(noise, verbose=0)
                self.epoch_disc_fake_loss.append(self.discriminator.train_on_batch(generated_data, np.ones(int(batch_size))))

            # freeze D and C
            self.discriminator.trainable = False
            for l in self.discriminator.layers:
                l.trainable = False

            noise = np.random.normal(loc=0.0, scale=1, size=(int(batch_size), self.latent_dim))
            self.epoch_gen_loss.append(self.combined.train_on_batch(noise, -np.ones(int(batch_size))))

            # Below code used solely for graphing when testing engine.py alone

            if(index % int(nb / 5) == 0):
               self.fake_data_list.append(self.gen_fake_data(300))

    def gen_fake_data(self, N = 1000):
        # Uses generator to generate fake data.

        fake_data = pd.DataFrame()

        for x in range(N):
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            gen_data = self.generator.predict(noise)
            gen_data = gen_data * self.col_max_array
            fake_data = fake_data.append(gen_data)

        return fake_data


def wasserstein_loss(y_true, y_pred):
    # Returns the result of the wasserstein loss function.

    return K.mean(y_true * y_pred)

# ALL CODE BELOW IS ONLY USED WHEN RUNNING ENGINE.PY ALONE AND IS NOT USED IN COLUMNS.PY
# THIS CODE'S ONLY PURPOSE IS FOR TROUBLESHOOTING
#__________________________________________________________________________________________________________________________________________


def show_gen_loss(gan):
    #Show generator loss value over iteration epochs.
    loss_plot = plt.figure(1)
    plt.plot(gan.epoch_gen_loss, c=(0,0,0))
    plt.title('Generator Loss over each training iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Generator Loss')
    loss_plot.show()

def real_vs_fake(real_data, fake_data_list, x_col, y_col):
    fig, axs = plt.subplots(5, sharex=False)
    fig.suptitle('Real and Fake Data Comparison')

    colors = ('blue', 'red')
    groups = ('fake', 'real')
    
    for pnum in range(len(fake_data_list)):
        fake_data = fake_data_list[pnum]
        fake_data_points = (fake_data[x_col], fake_data[y_col])
        real_data_points = (real_data[x_col], real_data[y_col])
        total_data = (fake_data_points, real_data_points)

        for total_data, color, group in zip(total_data, colors, groups):
            x, y = total_data
            axs[pnum].scatter(x, y, alpha = 0.2, c=color, edgecolors='none', s=30, label=group)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(loc=2)
    plt.show()

def export_fakes(gan, destination, N=1000):
    list_of_fakes = gan.gen_fake_data(N)
    export_csv = list_of_fakes.to_csv(destination)
    return export_csv

if __name__ == '__main__':
    cc_scaled_data, col_max, cc_data = process_csv("C:/Users/q1033821/Documents/VSCODE/datasets/heart-disease-uci/heart.csv")
    gan = GAN(cc_scaled_data, col_max)
    gan.train(epochs = 200, batch_size = 32, sample_size=300)

    export_fakes(gan, "C:/Users/q1033821/Documents/VSCODE/fake_data_factory/fake_data/fake_cc.csv", 300)
    show_gen_loss(gan)
    cc_sample_data = cc_data.sample(300)
    real_vs_fake(cc_sample_data, gan.fake_data_list, 'age', 'trestbps')
