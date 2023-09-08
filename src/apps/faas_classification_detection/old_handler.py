import logging
import array
import time
from timeit import default_timer as timer
import numpy as np
from resnet50_onnxrt import RunnerBase

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
PROFILE_NAME=f"{MODEL_NAME}-{BACKEND_NAME}"

PROFILE=SUPPORTED_PROFILES["defaults"]
MODEL_PROFILE, MODEL_PATH = get_profile_and_model_path(PROFILE_NAME)
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
if 'data-format' in PROFILE:
    image_format = PROFILE['data-format']
DS = wanted_dataset(data_path="/data/imagenet",
                    image_list=None,
                    name=PROFILE['dataset'],
                    image_format=image_format,
                    pre_process=pre_proc,
                    use_cache=0,
                    count=None, **kwargs)


def handle(event, _):
    BACKEND = get_backend(BACKEND_NAME)
    ml_start = timer()
    MODEL = BACKEND.load(MODEL_PATH, INPUTS, OUTPUTS)
    ml_end = timer()
    runner = RunnerBase(MODEL, DS, post_proc=post_proc,
                        max_batchsize=int(event['batch_size']))
    result_dict = {}
    runner.start_run(result_dict, False)
    responses, dl_time, inf_time = runner.run_samples(event['qs'])
    del MODEL

    return {
        'res': responses,
        'ml': round(ml_end - ml_start, 2),
        'dl': round(dl_time, 2),
        'inf': round(inf_time, 2),
    }


def main():
    qs = [{'idx': x, 'id': x+1} for x in range(24576)]
    event = {'qs': qs, 'batch_size': 16}
    start = timer()
    ret = handle(event, None)
    end = timer()
    print(ret)
    print(f"samples/s: {round(len(qs) / (end - start), 1)}")


if __name__ == '__main__':
    main()
