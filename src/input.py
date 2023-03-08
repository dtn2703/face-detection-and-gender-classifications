import threading
import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
from PIL import Image

from src.main_handle import load_yolo_model, yolo_predict, load_efficientnetb1_model, load_efficientnetb3_model, efficientnet_predict, load_mtcnn_model, mtcnn_predict, load_facecascade_model, facecascade_predict, load_svm_model, svm_predict, load_facenet_svm_model, facenet_svm_predict
from src.data import *




def webcam_input(face_detection_model_name, gender_recognition_model_name):
    st.header("Webcam Live Feed")

    class FaceGenderApplication(VideoTransformerBase):
        # _width = WIDTH
        _detection_model_name = face_detection_model_name
        _recognition_model_name = gender_recognition_model_name
        _detection_model = None
        _recognition_model = None

        def __init__(self) -> None:
            self._model_lock = threading.Lock()
            self._update_model()

        def update_model_name(self, detection_model_name, recognition_model_name):
            self._detection_model_name = detection_model_name
            self._recognition_model_name = recognition_model_name
            if (self._detection_model_name != detection_model_name) or (self._recognition_model_name != recognition_model_name):
                self._update_model()

        def _update_model(self):
            detection_model_path = detection_models_dict[self._detection_model_name]
            recognition_model_path = recognition_models_dict[self._recognition_model_name]
            print(detection_model_path)
            print(recognition_model_path)
            with self._model_lock:
                if self._detection_model_name == 'yolo':
                    self._detection_model = load_yolo_model(detection_model_path)
                elif self._detection_model_name == 'mtcnn':
                    self._detection_model = load_mtcnn_model()
                elif self._detection_model_name == 'Facecascade':
                    self._detection_model = load_facecascade_model(detection_model_path)

                if self._recognition_model_name == 'efficientnetb1': 
                    self._recognition_model = load_efficientnetb1_model(recognition_model_path)
                elif self._recognition_model_name == 'efficientnetb3': 
                    self._recognition_model = load_efficientnetb3_model(recognition_model_path)
                elif self._recognition_model_name == 'svm': 
                    self._recognition_model = load_svm_model(recognition_model_path)
                elif self._recognition_model_name == 'facenet-svm': 
                    self._recognition_model = load_facenet_svm_model(recognition_model_path)

        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            image = cv2.flip(image, 1)

            if self._detection_model == None:
                return image

            orig_h, orig_w = image.shape[0:2]

            # cv2.resize used in a forked thread may cause memory leaks
            input = np.asarray(Image.fromarray(image))

            with self._model_lock:
                if self._detection_model_name == 'yolo':
                    boxes = yolo_predict(input, self._detection_model)
                elif self._detection_model_name == 'mtcnn':
                    boxes = mtcnn_predict(input, self._detection_model)
                elif self._detection_model_name == 'Facecascade':
                    boxes = facecascade_predict(input, self._detection_model)

                for box in boxes: 
                    x1, y1, x2, y2 = box
                    sub_image = input[y1:y2, x1:x2]
                    if self._recognition_model_name in ['efficientnetb1', 'efficientnetb3']:
                        gender = efficientnet_predict(sub_image, self._recognition_model)
                    elif self._recognition_model_name == 'svm':
                        gender = svm_predict(sub_image, self._recognition_model)
                    elif self._recognition_model_name == 'facenet-svm':
                        gender = facenet_svm_predict(image, self._recognition_model)

                    cv2.rectangle(input, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(input, str(gender), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


            result = Image.fromarray(input.astype(np.uint8))
            return np.asarray(result.resize((orig_w, orig_h)))

    ctx = webrtc_streamer(
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        ),
        video_transformer_factory=FaceGenderApplication,
        key="neural-style-transfer",
    )
    if ctx.video_transformer:
        ctx.video_transformer.update_model_name(face_detection_model_name, gender_recognition_model_name)
