import cv2
import mediapipe as mp
import mouse
import keyboard
import time 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

wristPos = 0
sensitivity = 250
zoomCenteHandDistance = 0.55
handsDistance = None

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
      #detect wrist
      if results.multi_hand_landmarks[0]:
        if wristPos: 
          deltaX = results.multi_hand_landmarks[0].landmark[0].x - wristPos[0].x
          deltaY = results.multi_hand_landmarks[0].landmark[0].y - wristPos[0].y
          mouse.move(-deltaX*sensitivity, deltaY*sensitivity, absolute=False, duration=0.01)
        else: 
          keyboard.press('shift')
          mouse.press('middle')
        wristPos = results.multi_hand_landmarks[0].landmark
      if len(results.multi_handedness)==2:
        handsDistance = results.multi_hand_landmarks[0].landmark[20].x - results.multi_hand_landmarks[1].landmark[20].x
        zoomNeeded = handsDistance - zoomCenteHandDistance
        keyboard.release('shift')
        mouse.wheel(zoomNeeded)
        keyboard.press('shift')
        print(handsDistance)
    else: 
      mouse.release('middle')
      keyboard.release('shift')
      wristPos = 0
        #time.sleep(0.25)
    
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()