from keras.preprocessing.image import  ImageDataGenerator
import os

def getTrainData(data_path,img_cols,img_rows,batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=3,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(data_path,'train'),
        target_size=(img_cols,img_rows),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical"
    )
    return train_generator


def getTestData(data_path,img_cols,img_rows,batch_size):
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(data_path,'test'),
        target_size=(img_cols,img_rows),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )

    return test_generator




