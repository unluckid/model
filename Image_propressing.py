import cv2
import numpy as np
import os
import shutil

class Image_propressing:
    def __init__(self):
        #  폴더 경로
        self.video_path = "video/"  # 비디오 폴더
        self.nature_img_folder_path = "nature/"  # 전처리 이전 폴더
        self.model1_img_folder_path = "model1/"  # 기본 모델 폴더
        self.model2_img_folder_path = "model2/"  # 캐니 모델 폴더
        self.train_path = "train/"
        self.val_path = "val/"
        self.data = []  # 전처리 전 데이터셋

        self.label = ["_0", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9"]  # 10개의 이미지 폴더에 대한 값
        self.model_data1 = []  # 기본 이미지 모델 데이터셋
        self.model_data2 = []  # 캐니 이미지 모델 데이터셋
        self.valimgs = []    # 폴더 내 이미지 파일 이름을 저장할 리스트
        self.trainimgs = []  # 폴더 내 이미지 파일 이름을 저장할 리스트
        self.train_save = 3
        self.val_save = 11
    def reset_folder(self):
        shutil.rmtree(self.train_path)
        shutil.rmtree(self.val_path)
        print("폴더 초기화")
    def set_folder(self):
        os.makedirs(self.video_path, exist_ok=True)
        os.makedirs(self.train_path + self.nature_img_folder_path, exist_ok=True)
        os.makedirs(self.train_path + self.model1_img_folder_path, exist_ok=True)
        os.makedirs(self.train_path + self.model2_img_folder_path, exist_ok=True)
        os.makedirs(self.val_path + self.nature_img_folder_path, exist_ok=True)
        os.makedirs(self.val_path + self.model1_img_folder_path, exist_ok=True)
        os.makedirs(self.val_path + self.model2_img_folder_path, exist_ok=True)
        for label in self.label:                   # 10개의 하위 폴더 생성 코드 추가 작성
            os.makedirs(self.video_path + label, exist_ok=True)
            os.makedirs(self.train_path + self.nature_img_folder_path + label, exist_ok=True)
            os.makedirs(self.train_path + self.model1_img_folder_path + label, exist_ok=True)
            os.makedirs(self.train_path + self.model2_img_folder_path + label, exist_ok=True)
            os.makedirs(self.val_path + self.nature_img_folder_path + label, exist_ok=True)
            os.makedirs(self.val_path + self.model1_img_folder_path + label, exist_ok=True)
            os.makedirs(self.val_path + self.model2_img_folder_path + label, exist_ok=True)
        print("setting folder")

    def save_train(self):
        for label in self.label:
            for file_name in os.listdir(self.video_path + label):
                file_path = os.path.join(self.video_path + label, file_name)
                video_capture = cv2.VideoCapture(file_path) # 동영상 로드
                if not video_capture.isOpened():
                    print("error_00: " + label + ": 동영상 없음 ")
                frame_count, train_count, val_count = 0, 0, 0
                fps = video_capture.get(cv2.CAP_PROP_FPS)  # fps 설정
                train_save = int(fps * self.train_save)  # 훈련에 사용될 이미지 정리  3초 간격으로 작성
                val_save = int(fps * self.val_save)   # 훈련에 사용될 이미지 정리  11초 간격으로 작성
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break  # 프레임이 더 이상 없으면 반복문 종료
                    # 프레임 파일 이름 형식 지정 및 저장
                    if frame_count % train_save == 0:
                        frame_filename = os.path.join(self.train_path + self.nature_img_folder_path + label, f"{file_name}_{train_count:04d}.png")
                        frame = cv2.resize(frame, (240, 240))  # 이미지 크기 미리 정리
                        cv2.imwrite(frame_filename, frame)
                        train_count += 1
                    elif frame_count % val_save == 0:  # elif 이미지 중복 제거
                        frame_filename = os.path.join(self.val_path + self.nature_img_folder_path + label, f"{file_name}_{val_count:04d}.png")
                        frame = cv2.resize(frame, (240, 240))  # 이미지 크기 미리 정리
                        cv2.imwrite(frame_filename, frame)
                        val_count += 1
                    frame_count += 1
                    # 자원 해제
                video_capture.release()
            print("프레임 분할 완료!")

    # 이미지의 폴더 패스를 받고 이를 self.imgs 에 저장한다.
    def set_img(self, model): #이미지 이름을 모두 따서 self.imgs에 저장해둠
        self.valimgs = []
        self.trainimgs = []
        for label in self.label:
            for img in os.listdir(self.val_path + model + label):
                if os.path.splitext(img)[1].lower() in {".png", ".jpg"}:
                    self.valimgs.append([self.val_path, model, label, img])
                else:
                    print("error_01: 이미지 전처리 에러")
            print("이미지 전처리  패스 설정 완료")
        for label in self.label:
            for img in os.listdir(self.train_path + model + label):
                if os.path.splitext(img)[1].lower() in {".png", ".jpg"}:
                    self.trainimgs.append([self.train_path, model, label, img])
                else:
                    print("error_01: 이미지 전처리 에러")
            print("이미지 전처리  패스 설정 완료")

    def made_model1_dataset(self):
        self.set_img(self.nature_img_folder_path)
        for path in self.valimgs:
            image = cv2.imread(path[0]+path[1]+path[2]+"/"+path[3], cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            for i in range(12):
                r_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), i * 30, 1.0)
                r_img = cv2.warpAffine(image, r_matrix, (width, height))
                cv2.imwrite(self.val_path+self.model1_img_folder_path+path[2]+"/"+str(i)+path[3], r_img)
        for path in self.trainimgs:
            image = cv2.imread(path[0]+path[1]+path[2]+"/"+path[3], cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            for i in range(12):
                r_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), i * 30, 1.0)
                r_img = cv2.warpAffine(image, r_matrix, (width, height))
                cv2.imwrite(self.train_path+self.model1_img_folder_path+path[2]+"/"+str(i)+path[3], r_img)
        # return self.model_data1    # data리스트 반환

    def made_model2_dataset(self):
        self.set_img(self.model1_img_folder_path)
        for path in self.valimgs:
            image = cv2.imread(path[0] + path[1] + path[2]+"/"+path[3], cv2.IMREAD_GRAYSCALE)
            img_c = cv2.Canny(image, 150, 240)
            cv2.imwrite(path[0]+self.model2_img_folder_path+path[2]+"/"+path[3], img_c)
        for path in self.trainimgs:
            image = cv2.imread(path[0] + path[1] + path[2] + "/" + path[3], cv2.IMREAD_GRAYSCALE)
            img_c = cv2.Canny(image, 150, 240)
            cv2.imwrite(path[0] + self.model2_img_folder_path + path[2]+"/" + path[3], img_c)

if __name__ == '__main__':
    IMGP = Image_propressing()
    IMGP.reset_folder()  # 폴더 초기화
    IMGP.set_folder()
    IMGP.save_train()
    IMGP.made_model1_dataset()
    IMGP.made_model2_dataset()
    IMGP.train_path = "train1/"
    IMGP.val_path = "val1/"
    IMGP.train_save = 4
    IMGP.val_save = 13
    IMGP.set_folder()
    IMGP.save_train()
    IMGP.made_model1_dataset()
    IMGP.made_model2_dataset()
