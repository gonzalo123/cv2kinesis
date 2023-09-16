import cv2 as cv
import logging
import numpy as np
from ultralytics import YOLO

from lib.aws import aws_get_service
from settings import YOLO_MODEL, THRESHOLD, YOLO_CLASSES_TO_DETECT

logger = logging.getLogger(__name__)


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
