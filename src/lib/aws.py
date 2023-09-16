import boto3

from settings import AWS_REGION, AWS_PROFILE_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

if AWS_PROFILE_NAME:
    session = boto3.Session(profile_name=AWS_PROFILE_NAME, region_name=AWS_REGION)
else:
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION)


def aws_get_service(service_name):
    return session.client(service_name)
