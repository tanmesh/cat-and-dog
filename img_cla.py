import numpy as np
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from prepare_data import prepare_data


def img_classi():
    x, y = prepare_data()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_size=1)
    n_train = len(x_train)
    n_val = len(x_val)
    batch_size = 16
    img_width = img_height = 50

    print("Building model..")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print("model builed...")

    print('Data augmentation...')
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    print('Preparing data for training and validation set...')
    train_generator = train_datagen.flow(np.array(x_train), y_train, batch_size=batch_size)
    validation_generator = val_datagen.flow(np.array(x_val), y_val, batch_size=batch_size)

    print('Training starts...')
    model.fit_generator(train_generator, steps_per_epoch=n_train,
                        epochs=30, validation_data=validation_generator, validation_steps=n_val)
    print('Training completed...')

    print('Savinf the model...')
    model.save_weights('model_wieghts.h5')
    model.save('model_keras.h5')
    print("Model saved...")

    print('Generating test data...')
    x_test, y_test = prepare_data()
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = val_datagen.flow(np.array(x_test), bsatch_size=batch_size)
    pred = model.predict_generator(test_generator, verbose=1)

    print(pred)


img_classi()
