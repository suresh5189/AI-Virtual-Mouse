import cv2
import numpy as np
import time
import autopy
import HandTrackingModule as htm

widthCamera, heightCamera = 640, 480
widthScreen, heightScreen = autopy.screen.size()
frameR = 100 # Frame Reduction
smoothening = 7

previousTime = 0
previousLocationX, previousLocationY = 0, 0
currentLocationX, currentLocationY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3,widthCamera)
cap.set(4,heightCamera)

detector = htm.handDetection(maxHands=1)

while True:
    # Find Hand Landmarks
    success, img = cap.read()
    img = detector.findhands(img)
    lmList,bbox = detector.findPosition(img)
    
    # Get the tip of the Index and Middle Finger
    if len(lmList) != 0:
        print(lmList)
        
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        # print(x1,y1,x2,y2)
        
    # Check which Fingers are Up
        fingers = detector.fingersUp()
        print(fingers)
        cv2.rectangle(img,(frameR,frameR),(widthCamera-frameR,heightCamera-frameR),(255,0,255),2)
        
        # Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0 :

            # Convert Coordinates
            x3 = np.interp(x1,(frameR,widthCamera-frameR),(0,widthScreen))
            y3 = np.interp(y1,(frameR,heightCamera-frameR),(0,heightScreen))
        
            # Smooth Values
            currentLocationX = previousLocationX + (x3 - previousLocationX) / smoothening
            currentLocationY = previousLocationY + (y3 - previousLocationY) / smoothening
            
            # Move Mouse
            autopy.mouse.move(widthScreen - currentLocationX,currentLocationY)
            cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
            previousLocationX, previousLocationY = currentLocationX, currentLocationY
            
        # Both Index and Middle Fingers are Up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1 :
            
            # Finding Distance between the fingers
            length, img, lineInfo = detector.findDistance(8,12,img)
            print(length)
            
            # Click Mouse if distance is short
            if length < 40 :
                cv2.circle(img,(lineInfo[4],lineInfo[5]),10,(0,255,0),cv2.FILLED)
                autopy.mouse.click()  
    
    # Frame Rate
    currentTime = time.time()
    FPS = 1/(currentTime-previousTime)
    previousTime = currentTime
    
    cv2.putText(img,str(int(FPS)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    
    # Display
    cv2.imshow("Image",img)
    cv2.waitKey(1)