import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to read webcam")
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Blur mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Largest contour
        max_contour = max(contours, key=cv2.contourArea)

        # Ignore small contours
        if cv2.contourArea(max_contour) > 5000:

            # Draw contour
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 3)

            # Bounding box
            x, y, w, h = cv2.boundingRect(max_contour)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                "Hand Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # Show webcam
    cv2.imshow("Hand Detection", frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()