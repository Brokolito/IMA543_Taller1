# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Dense, Conv2D,concatenate, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras.layers import Flatten, add
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import math
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Importacion de imagenes y seccionar clases 
def extraer_imagenes(directorio):
    imagenes_train, clases_train = [], []
    imagenes_test, clases_test = [], []

    for tipo_datos in ['train', 'test']:
        ruta_tipo = os.path.join(directorio, tipo_datos)
        clases = sorted(os.listdir(ruta_tipo))  # Asegura orden consistente de clases

        for idx, clase in enumerate(clases):
            ruta_clase = os.path.join(ruta_tipo, clase)
            for foto in os.listdir(ruta_clase):
                ruta_foto = os.path.join(ruta_clase, foto)
                try:
                    img = image.load_img(ruta_foto, target_size=(48, 48), color_mode='grayscale')
                    img_array = np.array(img)

                    if tipo_datos == 'train':
                        imagenes_train.append(img_array)
                        clases_train.append(idx)
                    else:
                        imagenes_test.append(img_array)
                        clases_test.append(idx)
                except Exception as e:
                    print(f"Error al cargar {ruta_foto}: {e}")

    # Convertir a arrays y normalizar
    X_train = np.expand_dims(np.array(imagenes_train, dtype=np.float32) / 255.0, axis=-1)
    X_test = np.expand_dims(np.array(imagenes_test, dtype=np.float32) / 255.0, axis=-1)
    y_train = to_categorical(clases_train)
    y_test = to_categorical(clases_test)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = extraer_imagenes("FER")
input_shape = X_train.shape[1:]

batch_size = 32 
epochs = 200

data_augmentation = True
num_classes = 7

version = 2
n = 12
# Cálculo de la profundidad en dependencia del parámetro n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Nombre del modelo, profundidad y version
model_type = 'ResNet%dv%d' % (depth, version)

def lr_schedule(epoch):
    """Ajuste del Learning Rate
    Learning rate se reduce después de 80, 120, 160, 180 epocas.
    Esta función se llama automaticamente después de cada época durante
    el entrenamiento como parte de los callbacks.
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,activation='relu',batch_normalization=True,conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    Arguments:
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    Returns:
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size = kernel_size,
                  strides     = strides,
                  padding     = 'same',
                  kernel_initializer = 'he_normal',
                  kernel_regularizer = l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or 
    also known as bottleneck layer.
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, 
    the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, 
    while the number of filter maps is
    doubled. Within each stage, the layers have 
    the same number filters and the same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    Arguments:
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    Returns:
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 110 in [b])')
    # start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    
    # v2 performs Conv2D with BN-ReLU
    # on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,num_filters=num_filters_in,conv_first=True)

    # instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                # first layer and first stage
                if res_block == 0:  
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                # first layer but not first stage
                if res_block == 0:
                    # downsample
                    strides = 2 

            # bottleneck residual unit
            y = resnet_layer(inputs=x,num_filters=num_filters_in,kernel_size=1,strides=strides,
                             activation=activation,batch_normalization=batch_normalization,conv_first=False)
            y = resnet_layer(inputs=y,num_filters=num_filters_in,conv_first=False)
            y = resnet_layer(inputs=y,num_filters=num_filters_out,kernel_size=1,conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection
                # to match changed dims
                x = resnet_layer(inputs=x,num_filters=num_filters_out,kernel_size=1,
                                 strides=strides,activation=None,batch_normalization=False)
            x = add([x, y])

        num_filters_in = num_filters_out

    # add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(y)

    # instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth,num_classes=num_classes )
#else:
#    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr_schedule(0)),metrics=['acc'])
model.summary()

save_dir = os.path.join(os.getcwd(), 'saved_models_resnetv2')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_acc',verbose=2,save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    history=model.fit(X_test, y_train,batch_size=batch_size,epochs=epochs,
              validation_data=(X_test, y_test),shuffle=True,callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # this will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_test)

    steps_per_epoch =  math.ceil(len(X_test) / batch_size)
    # fit the model on the batches generated by datagen.flow().
    history=model.fit(x=datagen.flow(X_train, y_train, batch_size=batch_size),
              verbose=2,epochs=epochs,validation_data=(X_test, y_test),
              steps_per_epoch=steps_per_epoch,callbacks=callbacks)


# score trained model
scores = model.evaluate(X_test,y_test,batch_size=batch_size,verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

##-------------------
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('FunciÃ³n de PÃ©rdida')
plt.ylabel('Valor pÃ©rdida')
plt.xlabel('Ã‰pocas')
plt.legend(['Entrenamiento', 'Test'], loc='best')
plt.savefig('funcion_perdida_resnetv2.png')
##-------------------
# Verifica si es 'accuracy' o 'acc'
acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'

# Guardar exactitud
plt.figure()
plt.plot(history.history[acc_key])
plt.plot(history.history[val_acc_key])
plt.title('Exactitud del modelo')
plt.ylabel('Exactitud')
plt.xlabel('epocas')
plt.legend(['Entrenamiento', 'Validacion'], loc='best')
plt.savefig('funcion_exactitudmodelo_resnetv2.png')