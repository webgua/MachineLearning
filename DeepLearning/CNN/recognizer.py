"""
Keras CNN Models
"""
from keras import callbacks
img_rows,img_cols,img_channels=70,70,1
import abc
import os
import json



class Recognizer(abc.ABC):
    def __init__(self,train_data,validact_data,test_data=None):
        self.train_data= train_data
        self.validact_data = validact_data
        self.test_data =test_data
        self._model = None
        self._path_weights = None
        self._configure()

    @abc.abstractmethod
    def _configure(self):
        """
        configure thr recognizer
        """
        #int (Width of the input image)
        self.w_img = None
        # int (Height of the input image)
        self.h_img = None
        # int
        self.img_channels = None
        #bool (Whether to invert the input image)
        self.if_invert_img = None
        # int (Number of classes)
        self.num_classes = None
        # int (Batch Size)
        self.batch_size = None
        # str (Name(Version) of the recognizer)
        self.name_recognizer = None
        # bool (Whether to enable local debugging
        self.local_debug = None
        # bool (Whether to normalize input image so that mean is 0)
        self.normalize_img_mean0 = None
        # float (Percent of validaction data)
        self.percent_validaction = None

    @abc.abstractmethod
    def _create_model(self):
        """Create the model
        Set the self._model here
        :return:
        """
        self._model = None

    def _set_path_weights(self):
        path_weights = os.path.join(self.name_recognizer,'weights.hdfs')
        self._path_weights = path_weights

    def _train(self,epochs):
        self._load_weights()
        callback_checkpoints = callbacks.ModelCheckpoint(filepath=self._path_weights,
                                                         verbose=1,
                                                         save_best_only=True,
                                                         save_weights_only=True)

        dir_log_tensorboard = os.path.join(self.name_recognizer,'log_tensorboard')
        if not os.path.exists(dir_log_tensorboard):
            os.mkdir(dir_log_tensorboard)
        callback_tensorboard = callbacks.TensorBoard(log_dir=dir_log_tensorboard)

        self._model.fit_generator(self.train_data,
                        epochs = epochs,
                        verbose = 1,
                        validation_data = self.validact_data,
                        callbacks=[callback_checkpoints,callback_tensorboard])

    def _evaluate(self):
        self._load_weights()
        score = self._model.evaluate(self.test_data,verbose=0)
        path_test_results = os.path.join(self.dir_base,self.name_recognizer,'test_results.json')
        json.dump({'test_loss':score[0],'test_accutacy':score[1]},
                  open(path_test_results,'w'),indent=4,sort_keys=True)
        print('Test Loss:',score[0])
        print("Test accurecy",score[1])

    def run(self,epochs):
        self._create_model()
        self._train(epochs=epochs)
        if(self.test_data!=None):
            self._evaluate()

    def _load_weights(self):
        if self._path_weights is None:
            self._set_path_weights()
        if os.path.exists(self._path_weights):
            self._model.load_weights(self._path_weights)
            print('Wights are restored to ',self._path_weights)

# def build_LeNet(classes=200, include_top=True, input_shape=(img_rows, img_cols, img_channels)):
#     #img_input = Input(shape=input_shape)
#     model = Sequential()
#     model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),activation="relu",padding="valid",input_shape=input_shape,kernel_initializer='uniform'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(1024,activation="relu"))
#     model.add(Dense(classes,activation="softmax"))
#     print(model.summary())
#     return model
#
# def build_AlexNet(classes=200,include_top=True, input_shape=(img_rows,img_cols,img_channels)):
#     model=Sequential()
#     model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=input_shape, padding='valid', activation='relu',kernel_initializer='uniform'))
#     model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#     model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
#     model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#     model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
#     model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
#     model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(classes, activation='softmax'))
#     return model
#
# def build_ResNet30(classes=200,include_top=True,input_shape=(img_rows,img_cols,img_channels)):
#     model=Sequential()
#     model.add(ZeroPadding2D())