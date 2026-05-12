import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from math import hypot

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Screen size for virtual mouse
screen_width, screen_height = pyautogui.size()

# Drawing variables
canvas = None
prev_x, prev_y = 0, 0

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, c = img.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            landmarks = hand_landmarks.landmark

            # Get important finger points
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            thumb_tip = landmarks[4]

            # Convert to screen coordinates
            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)

            middle_x = int(middle_tip.x * w)
            middle_y = int(middle_tip.y * h)

            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            # Draw circles
            cv2.circle(img, (index_x, index_y), 10, (0,255,0), -1)
            cv2.circle(img, (middle_x, middle_y), 10, (255,0,0), -1)
            cv2.circle(img, (thumb_x, thumb_y), 10, (0,0,255), -1)

            # ==========================
            # Finger Counting
            # ==========================

            fingers = []

            # Thumb
            fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)

            # Fingers
            fingers.append(1 if landmarks[8].y < landmarks[6].y else 0)
            fingers.append(1 if landmarks[12].y < landmarks[10].y else 0)
            fingers.append(1 if landmarks[16].y < landmarks[14].y else 0)
            fingers.append(1 if landmarks[20].y < landmarks[18].y else 0)

            total_fingers = fingers.count(1)

            cv2.putText(
                img,
                f'Fingers: {total_fingers}',
                (20, 50),
                 cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            # ==========================
            # Gesture Recognition
            # ==========================

            gesture = "Unknown"

            if fingers == [0,0,0,0,0]:
                gesture = "Fist"

            elif fingers == [1,1,1,1,1]:
                gesture = "Open Palm"

            elif fingers == [1,0,0,0,0]:
                gesture = "Thumbs Up"

            cv2.putText(
                img,
                f'Gesture: {gesture}',
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                 1,
                (255,0,0),
                2
            )

            # ==========================
            # Virtual Mouse
            # ==========================

            mouse_x = np.interp(index_x, [0, w], [0, screen_width])
            mouse_y = np.interp(index_y, [0, h], [0, screen_height])

            pyautogui.moveTo(mouse_x, mouse_y)

            # Click when index and middle finger are close
            distance = hypot(middle_x - index_x, middle_y - index_y)

            if distance < 40:
                pyautogui.click()
                cv2.putText(img, 'CLICK', (20,150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 2)
                  # ==========================
            # Virtual Drawing
            # ==========================

            if total_fingers == 1:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = index_x, index_y

                cv2.line(canvas,
                         (prev_x, prev_y),
                         (index_x, index_y),
                         (255,0,255),
                         5)

                prev_x, prev_y = index_x, index_y

            else:
                prev_x, prev_y = 0, 0

            # ==========================
             # Volume Control Demo
            # ==========================

            thumb_index_distance = hypot(
                thumb_x - index_x,
                thumb_y - index_y
            )

            cv2.putText(
                img,
                f'Distance: {int(thumb_index_distance)}',
                (20,200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,0),
                2
            )

    # Merge drawing canvas
    img = cv2.add(img, canvas)

    cv2.imshow("Advanced Hand Gesture Project", img)
      # Press C to clear drawing
    key = cv2.waitKey(1)

    if key == ord('c'):
        canvas = np.zeros_like(img)

    # Press Q to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()