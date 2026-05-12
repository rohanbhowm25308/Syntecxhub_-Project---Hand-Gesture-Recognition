import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(
    "model/gesture_model.h5"
)

labels = ['fist', 'open_hand', 'palm', 'thumbs_up']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    success, frame = cap.read()

    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)

    roi = frame[100:300, 100:300]

    img = cv2.resize(roi, (64,64))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    class_index = np.argmax(prediction)

    gesture = labels[class_index]

    cv2.putText(
        frame,
        f'Gesture: {gesture}',
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()