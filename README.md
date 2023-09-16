## Real-time Object Detection and Streaming with Python, OpenCV, AWS Kinesis, and YOLOv8

I've got the following idea in my mind. I want to detect objects with YoloV8 and OpenCV. Here a simple example:

```python
import cv2 as cv
from ultralytics import YOLO
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level='INFO',
    datefmt='%d/%m/%Y %X')

logger = logging.getLogger(__name__)

X_RESIZE = 800
Y_RESIZE = 450

YOLO_MODEL="yolov8m.pt"
YOLO_CLASSES_TO_DETECT = 'bottle',
THRESHOLD = 0.80

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
```

However, my idea is slightly different. I intend to capture the camera stream, transmit it to AWS Kinesis, and subsequently, within a separate process, ingest this Kinesis stream for object detection or any other purpose using the camera frames. To achieve this, our initial step is to publish our OpenCV frames into Kinesis.
producer.py
```python
import logging

from lib.video import kinesis_producer
from settings import KINESIS_STREAM_NAME, CAM_URI, WIDTH

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level='INFO',
    datefmt='%d/%m/%Y %X')

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    kinesis_producer(
        kinesis_stream_name=KINESIS_STREAM_NAME,
        cam_uri=CAM_URI,
        width=WIDTH
    )
```

```python
def kinesis_producer(kinesis_stream_name, cam_uri, width):
    logger.info(f"start emitting stream from cam={cam_uri} to kinesis")
    kinesis = aws_get_service('kinesis')
    cap = cv.VideoCapture(cam_uri)
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                scale = width / frame.shape[1]
                height = int(frame.shape[0] * scale)

                scaled_frame = cv.resize(frame, (width, height))
                _, img_encoded = cv.imencode('.jpg', scaled_frame)
                kinesis.put_record(
                    StreamName=kinesis_stream_name,
                    Data=img_encoded.tobytes(),
                    PartitionKey='1'
                )

    except KeyboardInterrupt:
        cap.release()
```

Now our consumer.py

```python
import logging

from lib.video import kinesis_consumer
from settings import KINESIS_STREAM_NAME

logging.getLogger("ultralytics").setLevel(logging.WARNING)

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level='INFO',
    datefmt='%d/%m/%Y %X')

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    kinesis_consumer(
        kinesis_stream_name=KINESIS_STREAM_NAME
    )
```

To get the frames from kinesis stream we need to consider the shards. According to your needs you should 
change it a little bit.

```python
def kinesis_consumer(kinesis_stream_name):
    logger.info(f"start reading kinesis stream from={kinesis_stream_name}")
    kinesis = aws_get_service('kinesis')
    model = YOLO(YOLO_MODEL)
    response = kinesis.describe_stream(StreamName=kinesis_stream_name)
    shard_id = response['StreamDescription']['Shards'][0]['ShardId']
    while True:
        shard_iterator_response = kinesis.get_shard_iterator(
            StreamName=kinesis_stream_name,
            ShardId=shard_id,
            ShardIteratorType='LATEST'
        )
        shard_iterator = shard_iterator_response['ShardIterator']
        response = kinesis.get_records(
            ShardIterator=shard_iterator,
            Limit=10
        )

        for record in response['Records']:
            image_data = record['Data']
            frame = cv.imdecode(np.frombuffer(image_data, np.uint8), cv.IMREAD_COLOR)
            results = model.predict(frame)
            if detect_id_in_results(results):
                logger.info('Detected')

def detect_id_in_results(results):
    status = False
    for result in results:
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            probability = round(box.conf[0].item(), 2)
            if probability > THRESHOLD and class_id in YOLO_CLASSES_TO_DETECT:
                status = True
    return status
```

And that's all. Real-time Object Detection with Kinesis stream. However, computer vision examples are often simple and quite straightforward. On the other hand, real-world computer vision projects tend to be more intricate, involving considerations such as cameras, lighting conditions, performance optimization, and accuracy. But remember, this is just an example.
