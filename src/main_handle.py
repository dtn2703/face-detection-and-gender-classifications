import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models, transforms

from facenet_pytorch import MTCNN

from keras_facenet import FaceNet

cudnn.benchmark = True
plt.ion()



def load_svm_model(recognition_model_path):
  model = pickle.load(open(recognition_model_path, 'rb'))  
  return model

def svm_predict(image, model):
  class_name = ['female', 'male']
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.resize(image, (96, 64)).reshape((1, 96 * 64))
  gender = model.predict(image)
  return class_name[int(gender[0])]


def load_yolo_model(detection_model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=detection_model_path)
    model.eval()
    return model

def yolo_predict(image, model):
    # convert cv2 to PIL
    transfer_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transfer_image = Image.fromarray(transfer_image)

    # predict
    results = model([transfer_image], size=640)

    boxes = [result.tolist()[:4] for result in results.xyxy[0]]
    boxes = [list(map(int, box)) for box in boxes]
    return boxes


def load_efficientnetb1_model(recognition_model_path): 
    model = models.efficientnet_b1(pretrained=False)
    model.classifier = nn.Linear(model.classifier[1].in_features, 2)

    if not torch.cuda.is_available():
        checkpoint = torch.load(recognition_model_path, map_location=torch.device('cpu')) 
    else: 
        checkpoint = torch.load(recognition_model_path)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model

def load_efficientnetb3_model(recognition_model_path): 
    model = models.efficientnet_b3(pretrained=False)
    model.classifier = nn.Linear(model.classifier[1].in_features, 2)

    if not torch.cuda.is_available():
        checkpoint = torch.load(recognition_model_path, map_location=torch.device('cpu')) 
    else: 
        checkpoint = torch.load(recognition_model_path)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model

def efficientnet_predict(image, model): 
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    class_name = ['Female', 'Male']
    # convert cv2 to PIL
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Apply transform
    image = transform(image)

    image = image.reshape((1, 3, 224, 224))
    output = model(image).cpu().detach().numpy().argmax(axis=1)[0]
    
    return class_name[output]


def load_mtcnn_model():
    mtcnn = MTCNN()
    return mtcnn

def mtcnn_predict(image, model): 
    try: 
        result = model.detect(image)[0]
        result = [list(map(abs, i)) for i in result]
        result = [list(map(int, i)) for i in result]
    except: 
        return []
    return result


def load_facecascade_model(detection_model_path):
    face_cascade = cv2.CascadeClassifier(detection_model_path)
    return face_cascade

def facecascade_predict(image, model): 
    image = image.astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    boxes = [[i[0], i[1], i[0] + i[2], i[1] + i[3]] for i in boxes]
    return boxes


def load_facenet_svm_model(recognition_model_path):
    embedder = FaceNet()
    model = pickle.load(open(recognition_model_path, 'rb'))  
    return (embedder, model)

def facenet_svm_predict(image, model):
    class_name = ['male', 'female']
    model_facenet, model_svm = model
    required_size = (160, 160)
    image = cv2.resize(image, required_size)
    image = np.expand_dims(image, axis=0)

    embedding = model_facenet.embeddings(image)
    gender = model_svm.predict(embedding)

    return class_name[int(gender[0])]
