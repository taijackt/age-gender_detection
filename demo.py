# Environment: Mask

# for yolo face detector
from nn import *
from hyp import hyp

# main
import os
import cv2
import onnxruntime
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from utils.utils import non_max_suppression as nms
import time
from collections import deque
from queue import Queue, LifoQueue
import multiprocessing as mp
from skimage import io
import threading

point_num = hyp['point_num']

class age_gender_detector():
    def __init__(self, source, debug=False):
        self.device = torch.device("cuda")

        # attributes for face detection
        self.faceDetector = self.get_face_detector()
        self.conf_thres = 0.6
        self.iou_thres = 0.4

        # attribtues for age and gender classification
        self.ageClassifier = self.get_age_classifier()
        self.ageInputName = self.ageClassifier.get_inputs()[0].name

        self.genderClassifier = self.get_gender_classifier()
        self.genderInputName = self.genderClassifier.get_inputs()[0].name

        self.age_label = {0:"(0-15)",
                          1:"(15-25)",
                          2:"(15-25)",
                          3:"(25-35)",
                          4:"(25-35)",
                          5:"(35-45)",
                          6:"(35-45)",
                          7:"(<45)"}

        #self.test_img  = cv2.imread("/home/apptech/Desktop/")

        self.fps_queue = deque(maxlen=10)
        self.frame_queue = Queue(maxsize=10)
        
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        # attributes for video source
        self.height = self.cap.get(4)
        self.width = self.cap.get(3)

        self.getImageProcess = threading.Thread(target=self.get_image)
        self.getImageProcess.start()

    
    def get_image(self):
        while True:
            ret, frame = self.cap.read()
            if ret: 
                cv2.imwrite("./images/test.jpg",frame)
            else:
                continue;

    def get_face_detector(self):
        backone = mobilenetv3_small()
        net = DarknetWithShh(backone, hyp).to(self.device)
        weights = "./weights/mbv3_small_1_final.pt"
        net.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        net.eval()
        print("Load face detector successfully.")
        return net

    def get_age_classifier(self):
        sess = onnxruntime.InferenceSession("./models/age_net.onnx")
        print("Load age classifier successfully.")
        return sess

    def get_gender_classifier(self):
        sess = onnxruntime.InferenceSession("./models/gender_net.onnx")
        print("Load gender classifier successfully.")
        return sess

    def colorEqualize(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_img[:,:,2] =cv2.equalizeHist(hsv_img[:,:,2])
        hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return hsv_img
        
    def preprocessingForDetection(self, img):
        img = self.colorEqualize(img)
        img = Image.fromarray(img)
        #img = transforms.Resize((1080,1920))(img)
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)
        return img.to(self.device)

    def preprocessingForClassification(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227,227))
        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, axis=0)
        return img

    @torch.no_grad()
    def detect_face(self, imgTensor):
        return self.faceDetector(imgTensor)[0]

    def classify_face(self,det,frame):
        faces = []
        face_position = []

        # For loop to make a ndarray of faces
        for face in det:
            xmin ,ymin, xmax, ymax = face

            # Make the box larger
            xmin = xmin*0.98
            ymin = ymin*0.98
            xmax = xmax*1.02
            #ymax = ymax*1.02

            # prevent the box coordinate out of the shape of frame
            xmin = int(max(0, xmin))
            ymin = int(max(0, ymin))
            xmax = int(min(self.width, xmax))
            ymax = int(min(self.height, ymax))

            # save the face coordinate
            face_position.append([xmin,ymin,xmax,ymax])

            # crop the face out, preprocess it and save it in a list
            croppedFace = frame[ymin:ymax, xmin:xmax]
            croppedFace = self.preprocessingForClassification(croppedFace)
            faces.append(croppedFace)
        
        # batch of faces 
        faces = np.vstack(faces).astype(np.float32)

        # Start classification for age and gender
        start = time.time()
        genderPred = self.find_gender(faces)
        agePred = self.find_age(faces)
        print("Classification time:", time.time()-start)

        # Make the result in a np array
        result = np.hstack([face_position, genderPred.T,agePred.T])

        return result
        
    def find_age(self, imgNumpy):
        preds = self.ageClassifier.run(None, {self.ageInputName:imgNumpy})
        #print(preds)
        return np.argmax(preds, axis=2)

    def find_gender(self,imgNumpy):
        preds = self.genderClassifier.run(None, {self.genderInputName:imgNumpy})
        return np.argmax(preds, axis=2)

    def drawBoxes(self, results, frame):
        for pred in results:
            print(pred[5])
            label = "Male" if pred[4] == 0 else "Female"
            cv2.rectangle(frame, (pred[0], pred[1]), (pred[2], pred[3]), (0,255,0), 2)
            cv2.putText(frame, label+" : "+self.age_label[pred[5]], (pred[0],pred[1]), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2, cv2.LINE_AA)
            
    def run(self):
        while True:
            try:
                frame = io.imread("./images/test.jpg")
            except:
                continue
            
            # count time
            start = time.time()

            # preprocessing and detect face
            imgTensor = self.preprocessingForDetection(frame)
            preds = self.detect_face(imgTensor)

            # postprocess the result
            preds = nms(preds, self.conf_thres, self.iou_thres, multi_label=False, classes=0, agnostic= False,land=True ,point_num=point_num)

            #only 1 image, so just get the index 0 result
            preds = preds[0]

            # if the pred is not none, do classification on it
            if not preds is None: 
                det = preds[:,:4].cpu().numpy().astype(int)
                result = self.classify_face(det,frame)
                self.drawBoxes(result, frame)

            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fps = 1/ (time.time()-start)
            self.fps_queue.append(fps)
            cv2.putText(display_frame, "FPS:"+str(np.mean(self.fps_queue))[:4], (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Age Gender Detection", display_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                os.system("pkill -9 python")
                break

if __name__ == "__main__":
    engine = age_gender_detector(source="rtsp://root:pass@192.168.1.95/axis-media/media.amp?resolution=1280x720")
    #engine = age_gender_detector(source=-1)
    time.sleep(3)
    engine.run()


