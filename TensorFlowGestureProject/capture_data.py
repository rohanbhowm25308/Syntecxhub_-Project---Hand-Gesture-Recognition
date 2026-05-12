import cv2
import os

gesture_name = input("Enter gesture name: ")

path = f"dataset/{gesture_name}"

os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0

while True:

    success, frame = cap.read()

    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)

    roi = frame[100:300, 100:300]

    cv2.imshow("Capture Gesture", frame)

    key = cv2.waitKey(1)

    # Press S to save image
    if key == ord('s'):

        img_name = f"{path}/{count}.jpg"

        cv2.imwrite(img_name, roi)

        count += 1

        print(f"Saved {count}")

    # Press Q to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()