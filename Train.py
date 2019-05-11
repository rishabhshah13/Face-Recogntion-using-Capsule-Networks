import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
K.set_image_data_format('channels_last')
import Model
from datahandler import *


def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    #print(y_pred)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()
    
def margin_loss(y_true, y_pred):

    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))
    
def train(model, data):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks

    checkpoint = callbacks.ModelCheckpoint(save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (0.95 ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., lam_recon],
                  metrics={'capsnet': 'accuracy'})

    def train_generator(x, y, batch_size, shift_fraction=0.6):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction,
                                           horizontal_flip=True,
                                           rotation_range=30,
                                           zoom_range=0.6,
                                          vertical_flip=False)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])
            
    if(data_aug):
        print("AUG! \n")       
        model.fit([x_train, y_train], [y_train, x_train], batch_size=batch_size, epochs=epochs,
               validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[checkpoint, lr_decay])
    else:
        print("NOT AUG!\n")
       model.fit_generator(generator=train_generator(x_train, y_train, batch_size, shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / batch_size),
                        epochs=epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[checkpoint, lr_decay]
                      
                        
    model.save_weights(save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % save_dir)


    return model
    
    
    
batch_size = 1
epochs = 150
lam_recon = 0.5
num_routing = 5
shift_fraction = 0.1
debug = 0
save_dir = 'result'
weights = None
lr = 0.012
data_aug = True

if not os.path.exists('result'):
    os.makedirs('result')
    

K.clear_session()
# load data
(x_train, y_train), (x_test, y_test) = load_data(min_faces_per_person = 50)

# define model
model, eval_model = Model.capsModel(input_shape=x_train.shape[1:],
                            n_class=len(np.unique(np.argmax(y_train, 1))),
                            num_routing=num_routing)
model.summary()

train(model=model, data=((x_train, y_train), (x_test, y_test)))
test(model=eval_model, data=(x_test, y_test))

