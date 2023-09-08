import boto3
import io, os

s3_client = boto3.client(
    's3',
    aws_access_key_id="YOUR_aws_access_key_id",
    aws_secret_access_key="YOUR_aws_sercret_access_key_id"
)

#prefix = "/home/ubuntu/serverless-gpus/src/apps/mlperf/data/ILSVRC2012_img_val"

prefix = "/home/ubuntu/serverless-gpus/src/apps/mlperf/inference/vision/classification_and_detection/preprocessed/imagenet/NCHW"
with open("val_map_2048.txt") as f:
    for line in f:
        l = line.strip()
        words = l.split(" ")
        name, label = words[0].strip(), words[1].strip()
        name += ".npy"
        path = os.path.join(prefix, name)
        s3_client.upload_file(path, "hf-dgsf", f"resnet/inputs/{name}")

