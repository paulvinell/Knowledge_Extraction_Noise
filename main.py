# Start: code that stops warnings (not very good if something is actually wrong)
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# End: code that stops warnings

import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pickle
from pprint import pprint
from resnet import resnet_v2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from steps import *

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32')
    x_train = (x_train - 127.5) / 127.5
    x_test = x_test.astype('float32')
    x_test = (x_test - 127.5) / 127.5

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)

    early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True)

    teacher = create_model(x_train[0].shape)
    if not load_network(teacher, "Teacher"):
        teacher.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=50, verbose=1, shuffle=True, callbacks=[early_stop])
        save_network(teacher, "Teacher")

    student = create_model(x_train[0].shape)

    teacher.evaluate(x_test, y_test)
    student.evaluate(x_test, y_test)

    for i in range(1, 25):
        print("Round {}".format(i))
        for j in range(30):
            print("Batch {}/{}".format(j+1, 30))
            train_adversarial_noise(x_train[0].shape, student, teacher)

        student.evaluate(x_val, y_val)

    student.evaluate(x_test, y_test)

def create_model(input_shape):
    model = Sequential()
    model.add(resnet_v2(input_shape, 11))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

def save_network(model, part_string):
    if not os.path.exists("savedmodel"):
        os.mkdir("savedmodel")

    print("Attempting to save part {}...".format(part_string))
    try:
        model.save_weights('savedmodel/{}.h5'.format(part_string))

        print("Successfully saved part {}".format(part_string))
    except:
        print("Failed to save part {}".format(part_string))

def load_network(model, part_string):
    if not os.path.exists("savedmodel"):
        print("Could not find model map '/savedmodel'")
        return False

    print("Attempting to load part {}...".format(part_string))
    try:
        model.load_weights('savedmodel/{}.h5'.format(part_string))

        print("Successfully loaded part {}".format(part_string))
    except:
        print("Failed to load part {}".format(part_string))
        return False

    return True

if __name__ == "__main__":
    main()
