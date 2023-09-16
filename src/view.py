import cv2 as cv
from ultralytics import YOLO
import logging

from settings import YOLO_MODEL, YOLO_CLASSES_TO_DETECT, THRESHOLD

logging.getLogger("ultralytics").setLevel(logging.WARNING)

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level='INFO',
    datefmt='%d/%m/%Y %X')

logger = logging.getLogger(__name__)

X_RESIZE = 800
Y_RESIZE = 450

cap = cv.VideoCapture(0)
model = YOLO(YOLO_MODEL)
while True:
    _, original_frame = cap.read()
    frame = cv.resize(original_frame, (X_RESIZE, Y_RESIZE), interpolation=cv.INTER_LINEAR)

    results = model.predict(frame)

    color = (0, 255, 0)
    for result in results:
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            status = id == class_id
            cords = box.xyxy[0].tolist()
            probability = round(box.conf[0].item(), 2)

            if probability > THRESHOLD and class_id in YOLO_CLASSES_TO_DETECT:
                logger.info(f"{class_id} {round(probability * 100)}%")
                x1, y1, x2, y2 = [round(x) for x in cords]
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv.imshow(f"Cam", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
