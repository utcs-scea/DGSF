import argparse
import pickle
from timeit import default_timer as timer
import boto3, io, sys

s3_client = boto3.client(
    's3',
    aws_access_key_id="YOUR_aws_access_key_id",
    aws_secret_access_key="YOUR_aws_sercret_access_key_id"
)

class squad_v1():
    def __init__(self, inputio):
        #with open(cache_path, 'rb') as cache_file:
        #    self.eval_features = pickle.load(cache_file)
        self.eval_features = pickle.load(inputio)

    def get_features(self, sample_id):
        return self.eval_features[sample_id]


def main():
    start = timer()
    input_start = timer()

    from onnxruntime_SUT import get_onnxruntime_sut
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--count', default=512, type=int)  # default=24576

    #download inputs
    inputio = io.BytesIO()
    s3_client.download_fileobj("hf-dgsf", "bert/eval_features.pickle", inputio)
    inputio.seek(0)
    DS = squad_v1(inputio)
    input_end = timer()

    #download model
    modeldl_start = timer()
    modelio = io.BytesIO()
    s3_client.download_fileobj("hf-dgsf", "bert/model.onnx", modelio)
    modelio.seek(0)
    modeldl_end = timer()

    #load model
    modelload_start = timer()
    args = parser.parse_args()
    batchsize = args.batchsize
    qs = [{'idx': x, 'id': x+1} for x in range(args.count)]
    sut = get_onnxruntime_sut(model_path=modelio,
                              quantized=False,
                              profile=False,
                              batchsize=batchsize)
    modelload_end = timer()

    #exec
    exec_start = timer()
    responses = sut.issue_queries(DS, qs)
    end = timer()

    del sut

    ret = {
        "download_input": round((input_end-input_start)*1000, 2),
        "download_model": round((modeldl_end-modeldl_start)*1000, 2),
        "load_model": round((modelload_end-modelload_start)*1000, 2),
        "execution": round((end-exec_start)*1000, 2),
        "end-to-end": round((end-input_start)*1000, 2)
    }

    import json
    print(">!!"+json.dumps(ret))
    sys.stdout.flush()

    #import os
    #print("segfaulting to exit..")
    #os.kill(os.getpid(),11)

if __name__ == '__main__':
    main()