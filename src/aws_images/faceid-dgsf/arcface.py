import os
import logging
from timeit import default_timer as timer
import numpy as np
import onnxruntime as rt
import json
from sklearn import preprocessing
import boto3, io

s3_client = boto3.client(
    's3',
    aws_access_key_id="YOUR_aws_access_key_id",
    aws_secret_access_key="YOUR_aws_sercret_access_key_id"
)


class ArcFace(object):
    __slots__ = ['sess', 'batch', 'input_name', 'label_name', 'mload_t',
                 'ctx', 'mod', 'framework']

    def onnxrt_init(self, modelio, cpu_threads: int, hw: str):
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = cpu_threads
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = \
            rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if "RUN_CPU" in os.environ:
            providers = ['CPUExecutionProvider']
        else:
            providers = [ ('CUDAExecutionProvider', {'gpu_mem_limit': int(3 * 1024 * 1024 * 1024), 
                    'do_copy_in_default_stream': True, 'arena_extend_strategy': 'kSameAsRequested' } )]
        
        start = timer()
        self.sess = rt.InferenceSession(modelio.read(), providers=providers,
                                        sess_options=sess_options)
        self.input_name = self.sess.get_inputs()[0].name
        self.label_name = self.sess.get_outputs()[0].name
        # input_data = np.zeros((batch, 3, 112, 112), dtype=np.float32)
        # self.sess.run([self.label_name], {self.input_name: input_data})
        self.mload_t = timer() - start

    def mx_init(self, rmpath: str, hw: str):
        start = timer()
        sym, arg_params, aux_params = mx.model.load_checkpoint(rmpath, 0)
        if hw == 'cpu':
            self.ctx = mx.cpu()
        elif hw == 'gpu':
            self.ctx = mx.gpu()
        elif hw == 'eia':
            self.ctx = mx.eia()
        else:
            raise Exception("hw can only be one of cpu, gpu and eia")

        data_names = [graph_input for graph_input in sym.list_inputs()
                      if graph_input not in arg_params and
                      graph_input not in aux_params]
        self.mod = mx.mod.Module(symbol=sym, data_names=data_names,
                                 context=self.ctx, label_names=None)
        self.mod.bind(for_training=False,
                      data_shapes=[('data', (self.batch, 3, 112, 112))],
                      label_shapes=None)
        self.mod.set_params(arg_params=arg_params, aux_params=aux_params,
                            allow_missing=True, allow_extra=True)
        self.mload_t = timer() - start

    def __init__(self, rmpath: str, batch: int, cpu_threads: int, hw: str,
                 framework: str = 'onnxrt'):
        """
        hw: one of cpu, gpu and eia(AWS elastic inference)
        framework: one of onnxrt and mx(AWS elastic inference only
                   supports tensorflow, mxnet and pytorch)
        """
        self.batch = batch
        if framework == 'onnxrt':
            self.onnxrt_init(rmpath, cpu_threads, hw)
        elif framework == 'mx':
            self.mx_init(rmpath, hw)
        else:
            raise Exception("framework should be either onnxrt or mx")
        self.framework = framework

    def onnxrt_inference(self, input_data):
        assert input_data.shape[0] == self.batch
        pred = self.sess.run(
            [self.label_name], {self.input_name: input_data})
        return pred[0]

    def mx_inference(self, input_data):
        assert input_data.shape[0] == self.batch
        in_data_mx = nd.array(input_data, ctx=self.ctx)
        db = mx.io.DataBatch(data=(in_data_mx,))
        self.mod.forward(db, is_train=False)
        pred = self.mod.get_outputs()[0].asnumpy()
        return pred

    def inference(self, input_data):
        if self.framework == 'onnxrt':
            return self.onnxrt_inference(input_data)
        elif self.framework == 'mx':
            return self.mx_inference(input_data)

    def inference_batch(self, all_faces):
        num_faces = len(all_faces)
        in_data = np.zeros((self.batch, 3, 112, 112), dtype=np.float32)
        embedding_arr = []
        iteration = (num_faces-1) // self.batch + 1
        for i in range(iteration):
            tmp = all_faces[i*self.batch:i*self.batch+self.batch]
            in_data[0:len(tmp), :] = tmp
            res = self.inference(in_data)
            embedding = preprocessing.normalize(res)
            embedding_arr.append(embedding)
        embedding_arr = np.concatenate(embedding_arr, axis=0)
        return embedding_arr


def main():
    from dataset import LFWDataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, required=True)
    args = parser.parse_args()

    # ds = LFWDataset(args.dataset_path, args.pairs_path)
    # if args.count is not None:
    #     if args.count > len(ds.nameLs):
    #         count = ds.nameLs
    #     else:
    #         count = args.count
    #     id_list = [i for i in range(count)]
    # else:
    #     id_list = [i for i in range(0, len(ds.nameLs))]

    #model_load_end = timer()
    #nameL_imgs, nameR_imgs, flags = ds.get_samples(id_list)
    #read_data_end = timer()

    input_start = timer()
    image_list = {}
    flist = s3_client.list_objects(Bucket="hf-dgsf", Prefix="face_id/inputs")['Contents']

    def download_obj(key):
        img = io.BytesIO()
        obj = key['Key'].split("/")[-1]
        s3_client.download_fileobj("hf-dgsf", key['Key'], img)
        img.seek(0)
        image_list[obj] = np.load(img, allow_pickle=False)
        img.close()

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as executor:
        for key in flist:
            executor.submit(download_obj, key)

    print(f"downloaded {len(image_list.keys())} images")
    
    img_pairs_io = io.BytesIO()
    s3_client.download_fileobj("hf-dgsf", "face_id/input_pairs.txt", img_pairs_io)
    img_pairs_io.seek(0)
    img_pairs = img_pairs_io.read().decode('UTF-8')

    nameL_imgs = []
    nameR_imgs = [] 
    flags = []
    for line in img_pairs.splitlines():
        l,r,flag = line.split(",")
        nameL_imgs.append(image_list[l])
        nameR_imgs.append(image_list[r])
        flags.append(flag.strip() == "True")

    input_end = timer()

    modeldl_start = timer()
    modelio = io.BytesIO()
    s3_client.download_fileobj("hf-dgsf", "face_id/updated_arcfaceresnet100-8.onnx", modelio)
    modelio.seek(0)
    modeldl_end = timer()

    modelload_start = timer()
    model = ArcFace(modelio, min(args.batchsize, len(nameL_imgs)), int(os.environ["CPU_NTHREADS"]), "gpu")
    modelload_end = timer()
    
    exec_start = timer()

    left_embeddings = model.inference_batch(nameL_imgs)
    right_embeddings = model.inference_batch(nameR_imgs)
    sim_arr = np.sum(np.multiply(left_embeddings, right_embeddings), 1)
    end = timer()
    #print(f"end to end: {round(end-start, 2)} s, model loading: {round(model_load_end-start, 2)} s, data loading: {round(read_data_end-model_load_end, 2)} s, inf: {round(end-read_data_end, 2)}")
    #print(f"# of samples: {len(id_list)}, sample/s: {round(len(id_list)/(end-read_data_end), 1)}")

    #print("flags: ", flags)
    #print("sim_arr: ", sim_arr)
    correct = 0
    for i, sim in enumerate(sim_arr[0:len(nameL_imgs)]):
        if sim > 0.5:
            if flags[i]:
                correct += 1
            #else:
            #    logging.debug("{} and {} should not match".format(
            #        ds.nameLs[id_list[i]], ds.nameRs[id_list[i]]))
        else:
            #if flags[i]:
            #    logging.debug("{} and {} should match".format(
            #        ds.nameLs[id_list[i]], ds.nameRs[id_list[i]]))
            #else:
            if not flags[i]:
                correct += 1
    print(f"{correct} / {len(nameL_imgs)} correct")

    end = timer()

    ret = {
        "download_input": round((input_end-input_start)*1000, 2),
        "download_model": round((modeldl_end-modeldl_start)*1000, 2),
        "load_model": round((modelload_end-modelload_start)*1000, 2),
        "execution": round((end-exec_start)*1000, 2),
        "end-to-end": round((end-input_start)*1000, 2)
    }

    import json
    print(">!!"+json.dumps(ret))

    #import os
    #print("segfaulting to exit..")
    #os.kill(os.getpid(),11)

if __name__ == '__main__':
    main()
