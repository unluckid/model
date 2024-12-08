import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier,BaggingClassifier
import keras_tuner
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.set_visible_devices([], 'GPU')  # 사용하고자 하는 GPU 장치 ID 설정

class Model:
    def __init__(self):
        self.models = []
        self.model  = None
        self.model1 = None
        self.model2 = None
        self.train_dir = "train/model1"
        self.val_dir = "val/model1"
        self.train_dir1 = "train/model2"
        self.val_dir1 = "val/model2"
        self.bagging_model1 = None
        self.bagging_model2 = None
        self.train_generator= None
        self.val_generator = None
        self.color ="rgb"
        self.gray = "gray"
    def setmodel1(self):
        self.model1 = models.Sequential()
        self.model1.add( layers.Conv2D(filters = 32,kernel_size = (3,3), activation= 'relu', input_shape=(240,240,3), strides = (1,1), padding = 'same')) #128
        self.model1.add( layers.BatchNormalization())
        self.model1.add( layers.MaxPooling2D(2,2))
        self.model1.add( layers.Conv2D(filters = 64,kernel_size =(3,3), activation='relu'))   #64
        self.model1.add( layers.BatchNormalization())
        self.model1.add( layers.MaxPooling2D(2,2))
        self.model1.add( layers.Conv2D(filters = 32,  kernel_size =(5,5), activation= 'relu'))   #32
        self.model1.add( layers.BatchNormalization())
        self.model1.add( layers.MaxPooling2D(2,2))
        self.model1.add( layers.Flatten())
        self.model1.add( layers.Dense(units=512, activation= 'relu'))
        self.model1.add( layers.Dropout(0.5))
        self.model1.add(layers.Dense(units=10, activation='softmax'))
        self.model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("모델1 구조 설정")
        return self.model1
    def setmodel2(self):
        self.model2 = models.Sequential()

        self.model2.add(
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(240, 240, 1),
                          strides=(1, 1), padding='same'))  # 128
        self.model2.add(layers.BatchNormalization())
        self.model2.add(layers.MaxPooling2D(2, 2))
        self.model2.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) # 64
        self.model2.add(layers.BatchNormalization())
        self.model2.add(layers.MaxPooling2D(2, 2))
        self.model2.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))  # 32
        self.model2.add(layers.BatchNormalization())
        self.model2.add(layers.MaxPooling2D(2, 2))
        self.model2.add(layers.Flatten())
        self.model2.add(layers.Dense(units=512, activation='relu'))
        self.model2.add(layers.Dropout(0.5))
        self.model2.add(layers.Dense(units=10, activation='softmax'))
        self.model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("모델2 구조 설정")
        return self.model2
    def made_data(self, train_dir, val_dir, color):
        train_datagen = ImageDataGenerator(
            rescale = 1./255, shear_range = 0.3, zoom_range = 0.3, horizontal_flip = True)# 학습시마다 변경된 이미지
        val_datagen = ImageDataGenerator(rescale = 1./255)
        if color == "rgb":
            self.train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(240, 240),
                batch_size=16,
                class_mode = 'categorical',
                color_mode='rgb')
            self.val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=(240, 240),
                batch_size=16,
                class_mode = 'categorical',color_mode='rgb') #binary
        elif color == "gray":
            self.train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(240, 240),
                batch_size=16,
                class_mode='categorical',
                color_mode='grayscale')
            self.val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=(240, 240),
                batch_size=16,
                class_mode='categorical', color_mode='grayscale')  # binary
    def history(self):
        plt.plot(self.history_model1.history['accuracy'])
        plt.plot(self.history_model1.history['val_accuracy'])
        plt.plot(self.history_model2.history['accuracy'])
        plt.plot(self.history_model2.history['val_accuracy'])
        plt.xlabel('Epoch')
        plt.xlabel('Accuracy')
        plt.legend(['Train_model1', 'Test_model1','Train_model2', 'Test_model2'], loc='upper left')
        plt.show()

    def set_models(self):
        self.models.append(self.model1)
        self.models.append(self.model2)
        print("모델 저장")
    def save_model(self, model,filepath):
        model.save(filepath)
        print(filepath + "모델 저장")
    def made_voting_model(self, model1, model2):
        self.model = [model1, model2]
    def made_model(self,model):
        if model == "natural":
            # 앙상블 모델 함수 활성화 추가 코드
            self.model1 = self.setmodel1()
            self.made_data(self.train_dir, self.val_dir, self.color) #model1
            self.history_model1 = self.model1.fit(self.train_generator, epochs=10, validation_data=self.val_generator,validation_steps=1)
            return self.model1
        elif model == "canny":
            self.model2 = self.setmodel2()
            self.made_data(self.train_dir1, self.val_dir1, self.gray) #model2
            self.history_model2 = self.model2.fit(self.train_generator, epochs=10, validation_data=self.val_generator, validation_steps=1)
            return self.model2
        self.history()

    def predict(self, model):
        pred = model.predict()
        return pred

if __name__ == '__main__':
    with tf.device('/GPU:0'):
        model = Model()
        model2 = model.made_model("canny")
        model.save_model(model2, "canny.h5")
        model1 = model.made_model("natural")
        model.save_model(model1, "model.h5")
        voting_model1 = model.made_voting_model( model1, model2)
    # 배깅
    """
    model.train_dir = "train1/model1"
    model.val_dir = "val1/model1"
    model.train_dir1 = "train1/model2"
    model.val_dir1 = "val1/model2"
    model3 = model.made_model("canny")
    model4 = model.made_model("natural")
    voting_model2 = model.made_voting_model()
    """

