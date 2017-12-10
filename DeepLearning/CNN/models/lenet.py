from DeepLearning.CNN.recognizer import Recognizer
from keras import  models,layers,losses,optimizers

class lenet_model(Recognizer):
    def _configure(self):
        """
                configure thr recognizer
                """
        self.w_img = 70
        self.h_img = 70
        self.img_channels = 1
        self.if_invert_img = True
        self.num_classes = 200
        self.batch_size = 64
        self.name_recognizer = 'Lenet'
        self.local_debug = False
        self.normalize_img_mean0 = False
        self.percent_validaction = 0.1

    def _create_model(self):
        """
        5*5*32  2*2 5*5*64  flatten 512 200
        :return:
        """
        input_shape = (self.h_img,self.w_img,self.img_channels)
        model = models.Sequential()
        model.add(layers.Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=input_shape,padding="valid",kernel_initializer='uniform'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.Conv2D(64,kernel_size=(5,5),activation="relu",padding='valid',kernel_initializer='uniform'))
        model.add(layers.MaxPool2D(pool_size=(2,2,)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512,activation='relu'))
        model.add(layers.Dense(self.num_classes,activation='softmax'))
        print(model.summary())
        model.compile(loss = losses.categorical_crossentropy,
                      optimizer=optimizers.Adadelta(),
                      metrics=['accuracy'])
        self._model = model




