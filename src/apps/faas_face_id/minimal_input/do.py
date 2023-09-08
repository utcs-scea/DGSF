import boto3
import io, os

s3_client = boto3.client(
    's3',
    aws_access_key_id="YOUR_aws_access_key_id",
    aws_secret_access_key="YOUR_aws_sercret_access_key_id"
)

uniq = set()

pairs = []

with open("face_id_list.txt") as f:
    for line in f:
        l = line.strip()
        words = l.split(",")

        x, y = words[0].strip(), words[1].strip()

        xx = x.split("/")[-1]
        yy = y.split("/")[-1]
        pairs.append(f"{xx},{yy},{words[2]}")

        uniq.add(x)
        uniq.add(y)


with open("input_pairs.txt", "w") as f:
    for p in pairs:
        #print(p)
        f.write(p+"\n")


# prefix = "/home/ubuntu/serverless-gpus/src/apps/faas_face_id_client/input_data/lfw_faces"
# for f in uniq:
#     path = os.path.join(prefix, f)
#     obj = f.split("/")[-1]

#     #print(path, "hf-dgsf", f"face_id/inputs/{obj}")
#     s3_client.upload_file(path, "hf-dgsf", f"face_id/inputs/{obj}")


