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
