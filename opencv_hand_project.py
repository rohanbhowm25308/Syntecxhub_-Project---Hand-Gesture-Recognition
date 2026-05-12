import cv2
import numpy as np

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Drawing canvas
canvas = None

while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Create canvas
    if canvas is None:
        canvas = np.zeros_like(frame)

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
            cv2.drawContours(
                frame,
                [max_contour],
                -1,
                (0, 255, 0),
                3
            )

            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(max_contour)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            # Center point
            cx = x + w // 2
            cy = y + h // 2

            cv2.circle(
                frame,
                (cx, cy),
                10,
                (0, 0, 255),
                -1
            )

            # =========================
            # Gesture Detection
            # =========================

            gesture = "Hand Detected"

            # Convex hull
            hull = cv2.convexHull(max_contour)

            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(max_contour)

            if contour_area != 0:

                solidity = float(contour_area) / hull_area

                # Improved gesture thresholds
                if solidity > 0.85:
                    gesture = "Fist"

                elif solidity > 0.65 and solidity <= 0.85:
                    gesture = "Palm"

                else:
                    gesture = "Open Hand"

                # Show solidity value
                cv2.putText(
                    frame,
                    f'Solidity: {solidity:.2f}',
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

            # Display gesture
            cv2.putText(
                frame,
                f'Gesture: {gesture}',
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # =========================
            # Virtual Drawing
            # =========================

            cv2.line(
                canvas,
                (cx, cy),
                (cx + 1, cy + 1),
                (255, 0, 255),
                5
            )

    # Merge drawing
    frame = cv2.add(frame, canvas)

    # Show webcam
    cv2.imshow("OpenCV Hand Project", frame)

    key = cv2.waitKey(1)

    # Clear drawing
    if key == ord('c'):
        canvas = np.zeros_like(frame)

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()