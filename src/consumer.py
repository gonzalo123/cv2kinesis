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
