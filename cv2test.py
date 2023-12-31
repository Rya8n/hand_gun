######## HAND GUN ALPHA VERSION 1 ########

import cv2
import time
import math
import pickle
import random
import pydirectinput
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=1, trackCon=0.8):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB = cv2.flip(imgRGB, 1)
        img = cv2.flip(img, 1)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        if self.maxHands == 1:
            keypoints = []
            landmarkList = []
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[0]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    keypoints.append([lm.x, lm.y, lm.z])
                    landmarkList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

            return keypoints, landmarkList
        else:
            keypoints0 = []
            landmarkList0 = []
            keypoints1 = []
            landmarkList1 = []
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[0]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    keypoints0.append([lm.x, lm.y, lm.z])
                    landmarkList0.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
                try:
                    myHand = self.results.multi_hand_landmarks[1]
                    for id, lm in enumerate(myHand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        keypoints1.append([lm.x, lm.y, lm.z])
                        landmarkList1.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
                except:
                    pass

            return keypoints0, landmarkList0, keypoints1, landmarkList1

class highAccuracyCallback(tf.keras.callbacks.Callback): # just in case
   def on_epoch_end(self, epoch, logs={}):
      if logs.get('accuracy') is not None and logs.get('accuracy') > 0.95:
         print("\nModel reached morea than 95.0% accuracy. Stopping training")
         self.model.stop_training = True

def create_model(numOfClasses):
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(512, activation= 'linear',input_dim=210),
      tf.keras.layers.Dense(512, activation= 'linear'),
      tf.keras.layers.Dense(256, activation= 'linear'),
      tf.keras.layers.Dense(256, activation= 'tanh'),
      tf.keras.layers.Dense(128, activation= 'selu'),
      tf.keras.layers.Dense(128, activation= 'selu'),
      tf.keras.layers.Dense(64, activation= 'selu'),
      tf.keras.layers.Dense(numOfClasses, activation='softmax'),
  ])
  model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-5), metrics=['accuracy'])
  return model

def isExist(theList, theSearched):
    idx = 0
    while idx != len(theList):
        if theSearched == theList[idx]:
            return idx
        idx = idx + 1
    return -1    

def equalize_data(x_dataset, y_dataset):
    unique_labels, label_counts = np.unique(y_dataset, return_counts=True)
    min_samples = min(label_counts)
    equalized_x = []
    equalized_y = []

    for label in unique_labels:
        label_indices = [i for i, y in enumerate(y_dataset) if y == label]
        selected_indices = random.sample(label_indices, min_samples)
        
        equalized_x.extend([x_dataset[i] for i in selected_indices])
        equalized_y.extend([y_dataset[i] for i in selected_indices])

    return np.array(equalized_x), np.array(equalized_y)

def stratifiedSplit(x_dataset, y_dataset, test_size): 
    x_train, x_test, y_train, y_test = train_test_split(
        x_dataset, y_dataset, test_size=test_size, stratify=y_dataset
    )

    return x_train, x_test, y_train, y_test

def testModel(loaded_model,x_test,y_test,loaded_labels):
    preds = []
    for predTargetIdx in range(len(x_test)):
        pred = loaded_model.predict(x_test[predTargetIdx:predTargetIdx+1])
        print(x_test[predTargetIdx:predTargetIdx+1].shape)
        print(pred)
        preds.append(np.argmax(pred))
    preds = np.array(preds)
    conf_matrix = confusion_matrix(y_test, preds, labels=range(len(loaded_labels)))
    return conf_matrix

def getConfMatrix(conf_matrix,loaded_labels):
    print("Confusion Matrix:")
    print("True/Predicted\t", end="")
    for label in loaded_labels:
        print(label, end="\t")
    print()

    for i, row_label in enumerate(loaded_labels):
        print(row_label, end="\t\t")
        for j in range(len(loaded_labels)):
            print(conf_matrix[i, j],"      ", end="\t")
        print()

def trainNewModel():
    detector = handDetector(detectionCon=1, maxHands=1)
    classList = []
    cursorClassList = [0,1,2,3,4,5]
    x_dataset = []
    y_dataset = []
    currentKey = ""

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        keypoints, landmarks = detector.findPosition(img, draw=False)

        if currentKey != "" and len(landmarks) != 0:
            classIdx = None
            if isExist(classList, currentKey) == -1:
                classList.append(currentKey)
                classIdx = len(classList) - 1
            else:
                classIdx = isExist(classList, currentKey)
            kptsDists = []
            idx1 = 0
            while idx1 != len(keypoints)-1:
                idx2 = idx1+1
                while idx2 != len(keypoints):
                    kptsDists.append(
                    math.sqrt((
                        ((keypoints[idx1][0] -  keypoints[idx2][0]) ** 2 ) +
                        ((keypoints[idx1][1] -  keypoints[idx2][1]) ** 2 ) +
                        ((keypoints[idx1][2] -  keypoints[idx2][2]) ** 2 )
                    ))
                    )
                    idx2 = idx2 + 1
                idx1 = idx1 + 1
            x_dataset.append(kptsDists)
            y_dataset.append(classIdx)
            cv2.putText(img, f'Please keep your hand in the frame', (0, 40), cv2.FONT_HERSHEY_PLAIN,
                        1.5, (255, 0, 0), 3)
            cv2.putText(img, str(currentKey), (20, 450), cv2.FONT_HERSHEY_PLAIN,
                        6, (255, 0, 0), 5)
        elif len(landmarks) == 0:
            currentKey = ""

        if currentKey == "":
            print("Please put your hand in front of the camera and then type in the desired key")
            print("or type in END to finish capturing")
            currentKey = input("Key: ")
            if currentKey == "END":
                break
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    x_dataset = np.array(x_dataset, dtype=np.float32)
    y_dataset = np.array(y_dataset, dtype=np.float32)
    print(y_dataset.shape,"unique data acquired")
    print("across",len(classList),"different handsigns")

    print("Training model...")
    x_dataset, y_dataset = equalize_data(x_dataset, y_dataset)
    x_train, x_test, y_train, y_test = stratifiedSplit(x_dataset, y_dataset, test_size=0.2)
    x_test, x_val, y_test, y_val = stratifiedSplit(x_test, y_test, test_size=0.5)

    callbacks = highAccuracyCallback()
    model = create_model(len(classList))
    history = model.fit(x=x_train, y=y_train, epochs=500, callbacks=[callbacks], validation_data=(x_val,y_val))

    model.save("decemberTESTNET.h5")
    pickle.dump(classList, open("labels_decemberTESTNET.dat", "wb"))
    pickle.dump(cursorClassList, open("labelsCursor_decemberTESTNET.dat", "wb"))
    print("Model saved succesfully!")

    conf_matrix = testModel(model,x_test,y_test,classList)
    getConfMatrix(conf_matrix, classList)

def useModel():
    detector = handDetector(detectionCon=1, maxHands=2)
    loaded_model = keras.models.load_model("decemberTESTNET.h5")
    loaded_labels = pickle.load(open("labels_decemberTESTNET.dat", "rb"))
    loaded_labelsCursor = pickle.load(open("labelsCursor_decemberTESTNET.dat", "rb"))

    hasTouchedDown0 = False
    touchDownPosX0 = -1
    touchDownPosY0 = -1

    hasTouchedDown1 = False
    touchDownPosX1 = -1
    touchDownPosY1 = -1

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        keypoints0, landmarks0,  keypoints1, landmarks1 = detector.findPosition(img, draw=False)

        if len(landmarks0) != 0:
            kptsDists0 = []
            idx1 = 0
            isMouse0 = False
            while idx1 != len(keypoints0)-1:
                idx2 = idx1+1
                while idx2 != len(keypoints0):
                    kptsDists0.append(
                    math.sqrt((
                        ((keypoints0[idx1][0] -  keypoints0[idx2][0]) ** 2 ) +
                        ((keypoints0[idx1][1] -  keypoints0[idx2][1]) ** 2 ) +
                        ((keypoints0[idx1][2] -  keypoints0[idx2][2]) ** 2 )
                    ))
                    )
                    idx2 = idx2 + 1
                idx1 = idx1 + 1
            kptsDists0 = np.array([kptsDists0], dtype=np.float32)
            pred0 = loaded_model.predict(kptsDists0)
            detectedKey0 = loaded_labels[pred0.argmax()]
            cv2.putText(img, str(detectedKey0), (20, 450), cv2.FONT_HERSHEY_PLAIN,
                    6, (255, 0, 0), 5)

            ####### THE CODE BELOW IS FOR DEMO ONLY #######
            ##### NEVER EVER USE IT FOR REAL GAMEPLAY #####
            print(detectedKey0)
            for chosenIdx in loaded_labelsCursor:
                if loaded_labels[chosenIdx] == detectedKey0:
                    isMouse0 = True
                    if hasTouchedDown0 == False:
                        hasTouchedDown0 = True
                        touchDownPosX0 = int(wScreen/wCam*keypoints0[4][0]*wCam)
                        touchDownPosY0 = int(hScreen/hCam*keypoints0[4][1]*hCam)
                    deltaX = touchDownPosX0 - int(wScreen/wCam*keypoints0[4][0]*wCam)
                    deltaY = touchDownPosY0 - int(hScreen/hCam*keypoints0[4][1]*hCam)
                    current_x, current_y = pydirectinput.position()
                    desiredX = current_x - deltaX
                    desiredY = current_y - deltaY
                    if desiredX < 0:
                        desiredX = 0
                    elif desiredX > wScreen:
                        desiredX = wScreen
                    if desiredY < 0:
                        desiredY = 0
                    elif desiredY > hScreen:
                        desiredY = hScreen
                    #pydirectinput.move(None, 0)
                    #pydirectinput.moveTo(current_x - deltaX, current_y - deltaY)

            if detectedKey0 == 'm1':
                pydirectinput.click(button='left')
            if detectedKey0 == 'm2':
                pydirectinput.click(button='right')
            if detectedKey0 == 'special':
                pydirectinput.keyDown('3')
                pydirectinput.keyUp('3')
                time.sleep(0.5)
                pydirectinput.click(button='left')


            if isMouse0 != True:
                hasTouchedDown0 = False
                touchDownPosX0 = -1
                touchDownPosY0 = -1

        if  len(landmarks1) != 0:
            kptsDists1 = []
            idx1 = 0
            isMouse1 = False
            while idx1 != len(keypoints1)-1:
                idx2 = idx1+1
                while idx2 != len(keypoints1):
                    kptsDists1.append(
                    math.sqrt((
                        ((keypoints1[idx1][0] -  keypoints1[idx2][0]) ** 2 ) +
                        ((keypoints1[idx1][1] -  keypoints1[idx2][1]) ** 2 ) +
                        ((keypoints1[idx1][2] -  keypoints1[idx2][2]) ** 2 )
                    ))
                    )
                    idx2 = idx2 + 1
                idx1 = idx1 + 1
            kptsDists1 = np.array([kptsDists1], dtype=np.float32)
            pred1 = loaded_model.predict(kptsDists1)
            detectedKey1 = loaded_labels[pred1.argmax()]
            cv2.putText(img, str(detectedKey1), (20, 410), cv2.FONT_HERSHEY_PLAIN,
                    6, (255, 0, 0), 5)

            if isMouse1 != True:
                hasTouchedDown1 = False
                touchDownPosX1 = -1
                touchDownPosY1 = -1

        ####### THE CODE ABOVE IS FOR DEMO ONLY #######
        ##### NEVER EVER USE IT FOR REAL GAMEPLAY #####

        cv2.imshow("Image", img)
        cv2.waitKey(1)

wCam, hCam = 1080, 720 # input cam size here
wScreen, hScreen = 1920, 1080 # input screen size here
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
while True:
    menuSelector = ""
    print("-------Hand Gun v0.1-------")
    print("MENU")
    print("1.LOAD")
    print("2.TRAIN")
    print("0.QUIT")

    menuSelector = input()
    if menuSelector == "1":
        useModel()
    elif menuSelector == "2":
        trainNewModel()
    elif menuSelector == "0":
        exit()
    