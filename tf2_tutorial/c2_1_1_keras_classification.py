import tensorflow as tf
from tensorflow import keras
import numpy as np

def main():
    print(tf.__version__)
    # load dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print('train images shape: {}'.format(train_images.shape))
    print('train labels shape: {}'.format(train_labels.shape))
    print('test images shape: {}'.format(test_images.shape))
    print('test labels shape: {}'.format(test_labels.shape))
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(10),
        keras.layers.Dropout()
    ])
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest acc: {}'.format(test_acc))

    probability_model = keras.Sequential([
        model,
        keras.layers.Softmax()
    ])
    predictions = probability_model.predict(test_images)
    print('\nPredict prob of the 1st test image: ', predictions[0])
    pred_cate = np.argmax(predictions[0])
    print('\nPredict cate of the 1st test image: ', pred_cate)
    pred_cate_name = class_names[pred_cate]
    print('\nPredict cate name of the 1st test image: ', pred_cate_name)


if __name__ == '__main__':
    main()
