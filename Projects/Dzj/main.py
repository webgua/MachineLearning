from IOs.Imgs_ios import getTestData,getTrainData
from DeepLearning.CNN.models.lenet import lenet_model


filepath  = "../../dataset/data-2"
train_data=getTrainData(data_path=filepath,img_cols=70,img_rows=70,batch_size=128)
test_data =getTestData( data_path=filepath, img_cols=70,img_rows=70,batch_size=128)

recognize = lenet_model(train_data=train_data,validact_data=test_data)
recognize.run(2)

print("123")