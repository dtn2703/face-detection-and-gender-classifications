import os


model_path = 'models'

# detection model
face_detection_models_weight = ['face_detection_yolov5s.pt', 'None', 'haarcascade_frontalface_default.xml', 'orther_path']
face_detection_model_name = ['yolo', 'mtcnn', 'Facecascade', 'orther_model']
detection_models_dict = {name: os.path.join(model_path, filee) for name, filee in zip(face_detection_model_name, face_detection_models_weight)}

# Recogntion
gender_recognition_models_weight = ['efficientnetB1.pt', 'efficientnetB3.pt', 'svm-recognition.pth', 'svm-facenet.pth', 'orther_path']
gender_recognition_model_name = ['efficientnetb1', 'efficientnetb3', 'svm', 'facenet-svm', 'orther_model']
recognition_models_dict = {name: os.path.join(model_path, filee) for name, filee in zip(gender_recognition_model_name, gender_recognition_models_weight)}
