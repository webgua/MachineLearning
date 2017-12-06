from keras import Input, regularizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,ZeroPadding2D
from keras.models import Sequential


img_rows,img_cols,img_channels=70,70,1

def build_LeNet(classes=200, include_top=True, input_shape=(img_rows, img_cols, img_channels)):
    #img_input = Input(shape=input_shape)
    model = Sequential()
    #filter卷积核的数目
    #activation 激活函数
    #kernel_size 卷积核的宽度和长度
    #kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器
    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),activation="relu",padding="valid",input_shape=input_shape,kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024,activation="relu"))
    model.add(Dense(classes,activation="softmax"))
    print(model.summary())
    return model

def build_AlexNet(classes=200,include_top=True, input_shape=(img_rows,img_cols,img_channels)):
    model=Sequential()
    model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=input_shape, padding='valid', activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    return model

def build_ResNet30(classes=200,include_top=True,input_shape=(img_rows,img_cols,img_channels)):
    model=Sequential()
    model.add(ZeroPadding2D())