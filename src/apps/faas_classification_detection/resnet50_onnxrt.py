import logging
import array
import time
from timeit import default_timer as timer
import argparse
import numpy as np
import boto3, io
import sys

s3_client = boto3.client(
    's3',
    aws_access_key_id="YOUR_aws_access_key_id",
    aws_secret_access_key="YOUR_aws_sercret_access_key_id"
)

from settings import SUPPORTED_DATASETS, SUPPORTED_PROFILES, \
    get_backend, Item, get_profile_and_model_path, get_img_format
MAX_CONCURRENCY = 4
MAX_POOL = 4
RETRY = 200

NANO_SEC = 1e9
MILLI_SEC = 1000

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

BACKEND_NAME = "onnxruntime"
MODEL_NAME = "resnet50"
PROFILE_NAME = f"{MODEL_NAME}-{BACKEND_NAME}"

PROFILE = SUPPORTED_PROFILES["defaults"]
# "defaults": {
#         "dataset": "imagenet",
#         "backend": "tensorflow",
#         "cache": 0,
#         "max-batchsize": 32,
#     },
MODEL_PROFILE, _ = get_profile_and_model_path(PROFILE_NAME)
PROFILE.update(MODEL_PROFILE)
INPUTS = PROFILE.get('inputs', "")
if INPUTS:
    INPUTS = INPUTS.split(",")
OUTPUTS = PROFILE.get('outputs', "")
if OUTPUTS:
    OUTPUTS = OUTPUTS.split(",")
print("done initialization")

wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[PROFILE['dataset']]
image_format = get_img_format(BACKEND_NAME)
#NCHW
if 'data-format' in PROFILE:
    image_format = PROFILE['data-format']


class RunnerBase:
    def __init__(self, model, ds, post_proc=None, max_batchsize=128):
        self.ds = ds
        self.model = model
        self.max_batchsize = max_batchsize
        self.post_process = post_proc
        self.take_accuracy = False
        self.result_timing = []
        self.result_dict = {}

    def start_run(self, result_dict, take_accuracy):
        self.result_dict = result_dict
        self.result_timing = []
        self.take_accuracy = take_accuracy
        self.post_process.start()

    def run_one_item(self, qitem):
        # run the prediction
        processed_results = []
        try:
            results = self.model.predict({self.model.inputs[0]: qitem.img})
            processed_results = self.post_process(results, qitem.content_id, qitem.label, self.result_dict)
            if self.take_accuracy:
                self.post_process.add_results(processed_results)
                self.result_timing.append(time.time() - qitem.start)
        except Exception as ex:  # pylint: disable=broad-except
            #src = [self.ds.get_item_loc(i) for i in qitem.content_id]
            #log.error("thread: failed on contentid=%s, %s", src, ex)
            log.error("thread: failed on contentid=")
            # since post_process will not run, fake empty responses
            processed_results = [[]] * len(qitem.query_id)
        finally:
            response_array_refs = []
            response = []
            for idx, query_id in enumerate(qitem.query_id):
                response_array = array.array("B", np.array(processed_results[idx], np.float32).tobytes())
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append((query_id, bi[0], bi[1]))
            return response

    def run_samples(self, query_samples):
        idx = [q['idx'] for q in query_samples]
        query_id = [q['id'] for q in query_samples]
        response = []
        data_total_time = 0
        inf_time = 0
        if len(query_samples) < self.max_batchsize:
            lbeg = timer()
            self.ds.load_query_samples(idx)
            data, label = self.ds.get_samples(idx)
            lend = timer()
            response.extend(self.run_one_item(Item(query_id, idx,
                                                   data, label)))
            ulbeg = timer()
            self.ds.unload_query_samples(idx)
            ulend = timer()
            data_total_time += (lend - lbeg) + (ulend - ulbeg)
            inf_time += ulbeg - lend
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                lbeg = timer()
                self.ds.load_query_samples(idx[i:i+bs])
                data, label = self.ds.get_samples(idx[i:i+bs])
                lend = timer()
                response.extend(
                    self.run_one_item(Item(query_id[i:i+bs], idx[i:i+bs],
                                           data, label)))
                ulbeg = timer()
                self.ds.unload_query_samples(idx[i:i+bs])
                ulend = timer()
                data_total_time += (ulend - ulbeg) + (lend - lbeg)
                inf_time += ulbeg - lend
        return response, data_total_time, inf_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=16)
    args = parser.parse_args()
    
    input_start = timer()
    ds = wanted_dataset(data_path=None,
                        image_list=None,
                        name=PROFILE['dataset'],
                        image_format=image_format,
                        pre_process=pre_proc,
                        use_cache=0,
                        count=None, cache_dir=None, **kwargs)
    
    qs = [{'idx': x, 'id': x+1} for x in range(ds.total_count)]
    event = {'qs': qs, 'batch_size': args.batchsize}
    
    input_end = timer()


    backend = get_backend(BACKEND_NAME)

    #download model
    modeldl_start = timer()
    modelio = io.BytesIO()
    s3_client.download_fileobj("hf-dgsf", "resnet/resnet50_v1.onnx", modelio)
    modelio.seek(0)
    modeldl_end = timer()

    modelload_start = timer()
    model = backend.load(modelio, INPUTS, OUTPUTS)
    modelload_end = timer()

    exec_start = timer()
    runner = RunnerBase(model, ds, post_proc=post_proc,
                        max_batchsize=int(event['batch_size']))
    result_dict = {}
    runner.start_run(result_dict, False)
    responses, dl_time, inf_time = runner.run_samples(event['qs'])
    end = timer()
    # print(f'end to end: {round(end-start, 2)} s')
    # print(f'model loading: {round(ml_end - ml_start, 2)} s')
    # print(f'data loading {round(dl_time, 2)} s')
    # print(f'inf: {round(inf_time, 2)} s')
    # print(f"sample/s: {round(len(qs) / (end - start), 1)}")

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
    
    import os
    print("segfaulting to exit..")
    os.kill(os.getpid(),11)
    return 0

if __name__ == '__main__':
    main()
