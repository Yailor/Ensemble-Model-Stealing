from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np

base = 20

def augment(proportion):
    print("training set proportion={}".format(proportion))
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_other, X_aug, y_other, y_aug = train_test_split(X_train, Y_train, test_size=proportion,
                                                      random_state=42)

    train_datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest')

    num_row = 4
    num_col = 8
    num= num_row*num_col
    train_datagen.fit(X_aug.reshape(X_aug.shape[0], 28, 28, 1))
    aug_data = train_datagen.flow(X_aug.reshape(X_aug.shape[0], 28, 28, 1),
                                  y_aug.reshape(y_aug.shape[0], 1),
                                  batch_size=num, shuffle=False).x

    for i in range(20):
        aug_data_2 = train_datagen.flow(X_aug.reshape(X_aug.shape[0], 28, 28, 1),
                       y_aug.reshape(y_aug.shape[0], 1),
                       batch_size=num,shuffle=False).x
        aug_data = np.append(aug_data, aug_data_2, axis=0)

    print("augmentation finished")
    n_train_samples, nx_tr, ny_tr, nz_tr = aug_data.shape
    aug_data = aug_data.reshape((n_train_samples, nx_tr * ny_tr * nz_tr))

    return aug_data, X_aug, y_aug
