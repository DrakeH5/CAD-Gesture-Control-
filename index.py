import cv2
import mediapipe as mp
import mouse
import keyboard
import time 
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IndexPos = None
sensitivity = 500
verticalSensitivity = 150
handDistancePos = None
handsDistance = None
zoomSensitivity = 50


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      #draw to screen
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


      if len(results.multi_handedness)==2: #ZOOM MODE 
        thumbIndexDistance = math.sqrt(pow(results.multi_hand_landmarks[0].landmark[8].x - results.multi_hand_landmarks[0].landmark[4].x, 2)+pow(results.multi_hand_landmarks[0].landmark[8].x - results.multi_hand_landmarks[0].landmark[4].x, 2))
        if thumbIndexDistance > 0 and thumbIndexDistance < 0.01:
          handsDistance = math.sqrt(pow(results.multi_hand_landmarks[1].landmark[4].x - results.multi_hand_landmarks[0].landmark[4].x, 2) + pow(results.multi_hand_landmarks[1].landmark[4].y - results.multi_hand_landmarks[0].landmark[4].y, 2))
          if handDistancePos: 
            deltaHandsDistance = handsDistance - handDistancePos
            mouse.wheel(-deltaHandsDistance*zoomSensitivity)
          else: 
            keyboard.release('shift')
            IndexPos = None
          handDistancePos = handsDistance
      elif results.multi_hand_landmarks[0]: #ROTATE MODE 
        if IndexPos: 
          landMarks = results.multi_hand_landmarks[0]
          #rotating 
          deltaX = (landMarks.landmark[8].x-landMarks.landmark[0].x) - (IndexPos.landmark[8].x-IndexPos.landmark[0].x)
          deltaY = (landMarks.landmark[8].y-landMarks.landmark[0].y) - (IndexPos.landmark[8].y-IndexPos.landmark[0].y)
          deltaXY = math.sqrt(pow(deltaX, 2)+pow(deltaY, 2))*(deltaX/abs(deltaX))
          mouse.move(-deltaXY*sensitivity, 0, absolute=False, duration=0.01)
          #rotating up and down
          WristDeltaY = landMarks.landmark[0].y - IndexPos.landmark[0].y
          mouse.move(0, WristDeltaY*verticalSensitivity, absolute=False, duration=0.01)
        else: 
          keyboard.press('shift')
          mouse.press('middle')
          handDistancePos = None
        IndexPos = results.multi_hand_landmarks[0]
    
    
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      mouse.release('middle')
      mouse.release('shift')
      break
cap.release()
