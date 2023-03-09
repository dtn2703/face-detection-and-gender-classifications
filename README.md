# Face Detection and Gender Recognition

This application processes real-time face detection and gender recognition tasks, with the input being a computer camera.

**Detection:**
* YOLOv5
* MTCNN
* Haar Cascade (face)

**Gender recognition:**
* Efficientnet
* Linear kernel + SVM
* Facenet + SVM

**The technology used includes:**
* Python 
* Ubuntu
* Streamlit: building web applications for this projects.
* Thread in Python: a module used for creating and managing threads in a multi-threaded application. It allows the program to perform multiple tasks simultaneously, improving performance and user experience.

**Demo:**

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/k9OA-JBUQ1U/0.jpg)](https://www.youtube.com/watch?v=k9OA-JBUQ1U)

## 1. Download source code, models and install environment

```
# clone source code
git clone https://github.com/lynguyenminh/face-applicaation.git && cd face-applicaation

# install environment
pip install -r requirements.txt

# download weights for models
sh download_model.sh
```

## 2. Run application
```
streamlit run app.py
```
then access: [ https://localhost:8501](http://localhost:8501/) to use application.
## 3. References
[1]. https://www.whitphx.info/posts/20211231-streamlit-webrtc-video-app-tutorial/

[2]. https://github.com/whitphx/style-transfer-web-app
