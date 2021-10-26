import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import cvzone
from pynput.keyboard import Controller

cap=cv2.VideoCapture(0) #on webcam
cap.set(3,1280) #video width
cap.set(4,720) #video height

detector=HandDetector(detectionCon=0.8) #hand detection

keys=[["Q","W","E","R","T","Y","U","I","O","P",], #keyboard
      ["A","S","D","F","G","H","J","K","L",";"],
      ["Z","X","C","V","B","N","M",",",".","/"]]
finalText=""

keyboard=Controller()

def drawAll(img, buttonList):
     imgNew = np.zeros_like(img, np.uint8)
     for button in buttonList:
         x, y = button.pos
         cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                           20, rt=0)
         cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]),
                       (255, 0, 255), cv2.FILLED)
         cv2.putText(imgNew, button.text, (x + 40, y + 60),
                     cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

     out = img.copy()
     alpha = 0.3
     mask = imgNew.astype(bool)
     print(mask.shape)
     out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
     return out



class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text


buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))



while True:
    ret,frame=cap.read() #get frame
    frame=cv2.flip(frame,1) #mirror frame
    frame=detector.findHands(frame)
    lmList,bboxInfo=detector.findPosition(frame)
    frame=drawAll(frame,buttonList)

    if lmList:
        for button in buttonList:
            x,y=button.pos
            w,h=button.size

            if x<lmList[8][0]<x+w and y<lmList[8][1]<y+h:
                cv2.rectangle(frame, button.pos, (x + w, y + h), (175, 0, 175), cv2.FILLED)
                cv2.putText(frame, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                l,_,_=detector.findDistance(8,12,frame,draw=False)
                print(l)

                if l<30:
                    keyboard.press(button.text)
                    cv2.rectangle(frame, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    finalText+=button.text
                    sleep(0.15)

        cv2.rectangle(frame, (50,350), (700,450), (175, 0, 175), cv2.FILLED)
        cv2.putText(frame, finalText, (60,430), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    cv2.imshow("frame",frame)

cv2.destroyAllWindows()