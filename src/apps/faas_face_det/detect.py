"""
face detection using RetinaFace
"""
from math import ceil
from itertools import product
import os
from timeit import default_timer as timer
import numpy as np
import cv2
from data import cfg_mnet, cfg_re50  # noqa: E402
from utils.nms.nms import cpu_nms_wrapper  # noqa: E402
from detect_preprocess import preprocess_img, calculate_resize
import boto3
import io

s3_client = boto3.client(
    's3',
    aws_access_key_id="YOUR_aws_access_key_id",
    aws_secret_access_key="YOUR_aws_sercret_access_key_id"
)

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors, 10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    lms = np.concatenate(
        (priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
         priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
         priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
         priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
         priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
         ), 1)
    return lms


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    bboxs_ = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    bboxs_[:, :2] -= bboxs_[:, 2:] / 2
    bboxs_[:, 2:] += bboxs_[:, :2]
    return bboxs_


class PriorBox(object):
    __slots__ = ['min_sizes', 'steps', 'image_size', 'feature_maps']

    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step),
                              ceil(self.image_size[1]/step)]
                             for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1]
                                for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0]
                                for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = np.array(anchors)
        output = output.reshape(-1, 4)
        return output


class RetinaFace(object):
    __slots__ = ['sess', 'batch', 'input_name', 'label_names', 'mload_t',
                 'confidence_threshold', 'top_k', 'nms_threshold',
                 'keep_top_k', 'vis_thres', 'nms', 'cfg', 'net', 'ctx',
                 'model', 'framework', 'hw']

    def onnxrt_init(self, modelio, cpu_threads: int,
                    hw: str, no_sess: bool = False):
        import onnxruntime as rt
        sess_options = rt.SessionOptions()
        # sess_options.log_verbosity_level = 3
        # sess_options.enable_profiling = True
        sess_options.intra_op_num_threads = cpu_threads
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level =\
            rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CPUExecutionProvider']
        if hw == 'gpu':
            providers = [ ('CUDAExecutionProvider', {'gpu_mem_limit': int(13 * 1024 * 1024 * 1024), 'do_copy_in_default_stream': True, 
                    'arena_extend_strategy': 'kSameAsRequested' } )]
            #providers = ['CUDAExecutionProvider']

        start = timer()
        if no_sess:
            self.sess = None
        else:
            #self.sess = rt.InferenceSession(mpath, providers=providers,
            #                                sess_options=sess_options)
            
            self.sess = rt.InferenceSession(modelio.read() , providers=providers,
                                            sess_options=sess_options)

            self.input_name = self.sess.get_inputs()[0].name
            self.label_names = [out.name for out in self.sess.get_outputs()]
        # in_data = np.zeros((batch, 3, 640, 640), dtype=np.float32)
        # self.sess.run(self.label_names, {self.input_name: in_data})
        self.mload_t = timer() - start
        # self.run_options = rt.RunOptions()
        # self.run_options.log_severity_level = 0
        # self.run_options.log_verbosity_level = 3

    def mx_init(self, mpath: str, hw: str):
        from mxnet.contrib import onnx as onnx_mxnet
        from mxnet import gluon
        import mxnet as mx
        import warnings
        start = timer()
        sym, arg_params, aux_params = onnx_mxnet.import_model(mpath)
        if hw == 'cpu':
            self.ctx = mx.cpu()
        elif hw == 'gpu':
            self.ctx = mx.gpu()
        elif hw == 'eia':
            self.ctx = mx.eia()
        else:
            raise Exception("hw can only be one of cpu, gpu and eia")

        model_metadata = onnx_mxnet.get_model_metadata(mpath)
        data_names = [inputs[0]
                      for inputs in model_metadata.get('input_tensor_data')]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = gluon.nn.SymbolBlock(
                outputs=sym, inputs=mx.sym.var('data_0'))
        net_params = net.collect_params()
        for param in arg_params:
            if param in net_params:
                net_params[param]._load_init(arg_params[param], ctx=self.ctx)
        for param in aux_params:
            if param in net_params:
                net_params[param]._load_init(aux_params[param], ctx=self.ctx)
        net.hybridize()
        self.net = net
        self.mload_t = timer() - start

    def torch_init(self, mpath: str, hw: str, no_sess: bool):
        import torch
        if hw == 'eia':
            tmp_hw = 'cpu'
        else:
            tmp_hw = hw
        self.hw = hw
        if no_sess:
            self.model = None
        else:
            start = timer()
            self.model = torch.jit.load(
                mpath, map_location=torch.device(tmp_hw))
            self.mload_t = timer() - start

    def __init__(self, mpath, network, batch: int, cpu_threads: int,
                 hw: str, no_sess: bool = False, framework: str = 'onnxrt'):
        """
        hw: one of cpu, gpu and eia(AWS elastic inference)
        framework: one of onnxrt, mx, and torch(AWS elastic inference only
                   supports tensorflow, mxnet and pytorch)
        """
        self.batch = batch
        if framework == 'onnxrt':
            self.onnxrt_init(mpath, cpu_threads, hw, no_sess)
        elif framework == 'mx':
            self.mx_init(mpath, hw)
        elif framework == 'torch':
            self.torch_init(mpath, hw, no_sess)
        else:
            raise Exception("framework should be either onnxrt or mx or torch")

        self.framework = framework
        self.confidence_threshold = 0.02
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.vis_thres = 0.7

        self.nms = cpu_nms_wrapper(self.nms_threshold)

        if network == "mobile0.25":
            self.cfg = cfg_mnet
        elif network == "resnet50":
            self.cfg = cfg_re50
        else:
            raise Exception("network should be either mobile0.25 or resnet50")

    def mx_inference(self, in_data):
        from mxnet import nd
        assert in_data.shape[0] == self.batch
        start = timer()
        in_data_mx = nd.array(in_data, ctx=self.ctx)
        loc, conf, lms = self.net(in_data_mx)
        net_time = timer() - start
        return loc, conf, lms, net_time

    def onnxrt_inference(self, in_data):
        assert in_data.shape[0] == self.batch
        start = timer()
        loc, conf, lms = self.sess.run(self.label_names,
                                       {self.input_name: in_data}
                                       # , self.run_options
                                       )
        net_time = timer() - start
        return loc, conf, lms, net_time

    def torch_inference(self, in_data):
        import torch
        assert in_data.shape[0] == self.batch
        in_data = torch.from_numpy(in_data)
        start = timer()
        if self.hw == 'eia':
            with torch.no_grad():
                with torch.jit.optimized_execution(True,
                                                   {'target_device': 'eia:0'}):
                    loc, conf, lms = self.model(in_data)
        else:
            with torch.no_grad():
                loc, conf, lms = self.model(in_data)
        net_time = timer() - start
        return loc, conf, lms, net_time

    def inference(self, in_data):
        if self.framework == 'onnxrt':
            return self.onnxrt_inference(in_data)
        elif self.framework == 'mx':
            return self.mx_inference(in_data)
        elif self.framework == 'torch':
            return self.torch_inference(in_data)

    def inference_batch(self, all_imgs):
        num_imgs = len(all_imgs)
        in_data = np.zeros((self.batch, 3, 640, 640), dtype=np.int16)
        loc_b_arr, conf_b_arr, landmarks_b_arr = [], [], []
        sum_inf_time = 0
        iteration = (num_imgs-1) // self.batch + 1
        for i in range(iteration):
            tmp = all_imgs[i*self.batch:i*self.batch+self.batch]
            in_data[0:len(tmp), :] = tmp
            loc_b, conf_b, landmarks_b, inf_time = self.inference(in_data)
            loc_b_arr.append(loc_b)
            conf_b_arr.append(conf_b)
            landmarks_b_arr.append(landmarks_b)
            sum_inf_time += inf_time
        loc_b_arr = np.concatenate(loc_b_arr, axis=0)
        conf_b_arr = np.concatenate(conf_b_arr, axis=0)
        landmarks_b_arr = np.concatenate(landmarks_b_arr, axis=0)
        return loc_b_arr, conf_b_arr, landmarks_b_arr, sum_inf_time


def save_image(oname: str, dets, iname: str):
    num_face = 0
    img_raw = cv2.imread(iname)
    for b in dets:
        if len(b) == 0:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]),
                      (b[2], b[3]), (0, 0, 255), 2)
        num_face += 1
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        # eye left
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        # eye right
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        # nose
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        # mouse left
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        # mouse right
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
    # save image

    cv2.imwrite(oname, img_raw)
    return num_face


def post_process(loc, conf, landmarks, model, priors,
                 scale, scale1, resize: float):
    loc_squeezed = np.squeeze(loc, axis=0)
    boxes = decode(loc_squeezed, priors, model.cfg['variance'])
    boxes = boxes * scale / resize
    scores = np.squeeze(conf, axis=0)[:, 1]

    landmarks = decode_landm(np.squeeze(landmarks, axis=0),
                             priors, model.cfg['variance'])
    landmarks = landmarks * scale1 / resize

    # ignore low scores
    inds = np.where(scores > model.confidence_threshold)[0]
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:model.top_k]
    boxes = boxes[order]
    landmarks = landmarks[order]
    scores = scores[order]

    # do NMS(non maximum suppression)
    dets_ = np.hstack((boxes, scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
    keep = model.nms(dets_)
    dets_ = dets_[keep, :]
    landmarks = landmarks[keep]

    # keep top-K faster NMS
    dets_ = dets_[:model.keep_top_k, :]
    landmarks = landmarks[:model.keep_top_k, :]

    dets_ = np.concatenate((dets_, landmarks), axis=1)
    return dets_


def prepare_detection(cfg):
    height, width = 640, 640
    scale = np.array([width, height, width, height])
    scale1 = np.array([width, height, width, height,
                       width, height, width, height,
                       width, height])
    priorbox = PriorBox(cfg, image_size=(height, width))
    priors = priorbox.forward()
    return scale, scale1, priors


def detect_one_face(iname: str, oname: str, model, resize,
                    scale, scale1, priors, save_img=False):
    _img_raw = cv2.imread(iname)
    if _img_raw is None:
        return None, 0, 0
    img = np.float32(_img_raw)

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize,
                         interpolation=cv2.INTER_LINEAR)

    img -= (104, 117, 123)
    img = img.transpose((2, 0, 1))
    img_padded = np.zeros((3, 640, 640), dtype=np.float32)
    img_padded[:img.shape[0], :min(img.shape[1], 640),
               :min(img.shape[2], 640)] = img
    test = img_padded[np.newaxis, :, :, :]

    loc, conf, landmarks, net_time = model.inference(test)
    inf_time = net_time

    start = timer()
    dets_ = post_process(loc, conf, landmarks, model, priors, scale,
                         scale1, resize)
    pp_time = timer() - start
    if save_img:
        return (oname, dets_[dets_[:, 4] >= model.vis_thres],
                iname), inf_time, pp_time
    return (dets_[dets_[:, 4] >= model.vis_thres]), inf_time, pp_time


def detect_face(model, ipath: str, opath: str, pic_fmt: str,
                save_img=False):
    opath = os.path.expanduser(opath)
    if not os.path.exists(opath):
        os.makedirs(opath)
    flist = [(os.path.join(ipath, f),
              os.path.join(opath, f))
             for f in os.listdir(ipath)
             if f.endswith("." + pic_fmt)]
    img_fst = cv2.imread(flist[0][0])
    img_shape = img_fst.shape

    resize = calculate_resize(img_shape)
    scale, scale1, priors = prepare_detection(model.cfg)

    return_list = []

    sum_inf_time = 0
    sum_pp_time = 0
    for (iname_, oname_) in flist:
        ret, inf_time, pp_time = detect_one_face(iname_, oname_, model,
                                                 resize, scale,
                                                 scale1, priors, save_img)
        if ret is not None:
            sum_inf_time += inf_time
            sum_pp_time += pp_time
            return_list.append(ret)
    return return_list, sum_inf_time, sum_pp_time


def detect_face_batch(model, ipath: str, opath: str, pic_fmt: str,
                      save_img=False):
    d_start = timer()
    opath = os.path.expanduser(opath)
    if not os.path.exists(opath):
        os.makedirs(opath)
    flist = [(os.path.join(ipath, f),
              os.path.join(opath, f))
             for f in os.listdir(ipath)
             if f.endswith("." + pic_fmt)]

    scale, scale1, priors = prepare_detection(model.cfg)

    padded_image_list = []
    resize_list = []
    for (iname, _) in flist:
        img_raw_ = cv2.imread(iname)
        if img_raw_ is None:
            continue
        img_padded, resize = preprocess_img(img_raw_)
        padded_image_list.append(np.expand_dims(img_padded, axis=0))
        resize_list.append(resize)

    padded_image_list = np.concatenate(padded_image_list, axis=0)
    d_read_pre_t = timer() - d_start
    loc_b, conf_b, landmarks_b, net_time = \
        model.inference_batch(padded_image_list)
    length = len(padded_image_list)
    return_list = []
    sum_pp_time = 0
    for i in range(length):
        start_ = timer()
        dets_ = post_process(loc_b[i:i+1, :, :], conf_b[i:i+1, :, :],
                             landmarks_b[i:i+1, :, :], model, priors,
                             scale, scale1, resize_list[i])
        pp_time = timer() - start_
        if save_img:
            return_list.append((flist[i][1],
                                dets_[dets_[:, 4] >= model.vis_thres],
                                flist[i][0]))
        else:
            return_list.append((dets_[dets_[:, 4] >= model.vis_thres]))
        sum_pp_time += pp_time
    return return_list, d_read_pre_t, net_time, sum_pp_time


def get_image_list(image_list):
    images = []
    with open(image_list, "r") as fin:
        for s in fin:
            image_name = s.strip()
            images.append(image_name)
    return images


def get_image_batch(img_list, dataset_path):
    padded_img_list = []
    rewrite_list = []
    dl_t = 0
    for img in img_list:
        #img_raw = cv2.imread(os.path.join(dataset_path, img_name))
        img_raw = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)

        img_padded, rewrite = preprocess_img(img_raw)
        padded_img_list.append(img_padded)
        rewrite_list.append(rewrite)
    padded_img_list = np.array(padded_img_list)
    return padded_img_list, rewrite_list, dl_t


def main():
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='resnet50',
                        help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('-s', '--save_image', action="store_true",
                        default=True, help='save detection results')
    parser.add_argument("--pic_fmt", metavar="OUTPUT_PATH", type=str,
                        required=True)
    parser.add_argument("--bsize", metavar="BATCH_SIZE", type=int,
                        default=16)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpu_threads", metavar="CPU_THREADS", type=int,
                        default=1)
    parser.add_argument("--framework", metavar="FRAMEWORK", type=str,
                        default='onnxrt')
    parser.add_argument("--count", type=int, help="limit the length of image list")
    args = parser.parse_args()
    if args.gpu:
        hw = 'gpu'
    else:
        hw = 'cpu'
    
    #image_list = get_image_list(args.image_list)
    #if args.count is not None:
    #    image_list = image_list[:args.count]
    bs = args.bsize

    input_start = timer()

    from threading import Lock
    listlock = Lock()
    image_list = []
    flist = s3_client.list_objects(Bucket="hf-dgsf", Prefix="face_det/inputs")['Contents']

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def download_sublist(flist):
        imglist = []
        for key in flist:
            img = io.BytesIO()
            s3_client.download_fileobj("hf-dgsf", key['Key'], img)
            img.seek(0)
            image_list.append(img)

        listlock.acquire()
        image_list.extend(imglist)
        listlock.release()

    from math import ceil
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as executor:
        for chunk in chunks(flist, ceil(len(flist)/6)):
            executor.submit(download_sublist, chunk)
    print(f"downloaded {len(image_list)} images")
    input_end = timer()

    dets_arr = []
    length = len(image_list)
    dl_t = 0
    inf_t = 0
    post_proc_t = 0
    pre_proc_t = 0
    
    modeldl_start = timer()
    modelio = io.BytesIO()
    s3_client.download_fileobj("hf-dgsf", "face_det/updated_withpreprop_r50.onnx", modelio)
    modelio.seek(0)
    modeldl_end = timer()

    modelload_start = timer()
    model = RetinaFace(modelio, args.network, args.bsize,
                       args.cpu_threads, hw, framework=args.framework)
    modelload_end = timer()

    exec_start = timer()
    scale, scale1, priors = prepare_detection(model.cfg)
    for i in range(0, length, bs):
        sub_list = image_list[i:i+bs]
        padded_img_list, rewrite_list, dl = get_image_batch(sub_list, None)
        loc_b, conf_b, landmarks_b, inf = model.inference_batch(padded_img_list)
        pimg_list_len = len(padded_img_list)
        for j in range(pimg_list_len):
            dets_ = post_process(loc_b[j:j+1, :, :],
                                 conf_b[j:j+1, :, :], landmarks_b[j:j+1, :, :], model,
                                 priors, scale, scale1, rewrite_list[j])
            dets = dets_[dets_[:, 4] >= model.vis_thres]
            dets_arr.append((i+j, dets.tolist()))

    end = timer()
    # dets_arr_json = json.dumps(dets_arr)
    # print(dets_arr_json)

    ret = {
        "download_input": round((input_end-input_start)*1000, 2),
        "download_model": round((modeldl_end-modeldl_start)*1000, 2),
        "load_model": round((modelload_end-modelload_start)*1000, 2),
        "execution": round((end-exec_start)*1000, 2),
        "end-to-end": round((end-input_start)*1000, 2)
    }

    import json
    print(">!!"+json.dumps(ret))

    # print(f"end to end: {round(end-start, 2)}")
    # print(f"model loading: {round(ml_t, 2)}")
    # print(f"data loading: {round(dl_t, 2)}")
    # print(f"inf: {round(inf_t, 2)}")
    # print(f"inf+prep+post: {round(end-ml_end-dl_t, 2)}")
    # print(f"postp: {round(post_proc_t, 2)}")
    # print(f"prep: {round(pre_proc_t, 2)}")
    # print("samples/s: {}".format(round(length/(end - start), 1)))

    import os
    print("segfaulting to exit..")
    os.kill(os.getpid(),11)

if __name__ == '__main__':
    main()
