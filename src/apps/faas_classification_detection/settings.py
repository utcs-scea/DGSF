import time
import dataset
import imagenet
import coco
from dataclasses import dataclass

# pylint: disable=missing-docstring


# the datasets we support
SUPPORTED_DATASETS = {
    "imagenet":
        (imagenet.Imagenet, dataset.pre_process_vgg, dataset.PostProcessCommon(offset=-1),
         {"image_size": [224, 224, 3]}),
    "imagenet_mobilenet":
        (imagenet.Imagenet, dataset.pre_process_mobilenet, dataset.PostProcessArgMax(offset=-1),
         {"image_size": [224, 224, 3]}),
    "imagenet_pytorch":
        (imagenet.Imagenet, dataset.pre_process_imagenet_pytorch, dataset.PostProcessArgMax(offset=0),
         {"image_size": [224, 224, 3]}),
    "coco-300":
        (coco.Coco, dataset.pre_process_coco_mobilenet, coco.PostProcessCoco(),
         {"image_size": [300, 300, 3]}),
    "coco-300-pt":
        (coco.Coco, dataset.pre_process_coco_pt_mobilenet, coco.PostProcessCocoPt(False,0.3),
         {"image_size": [300, 300, 3]}),
    "coco-1200":
        (coco.Coco, dataset.pre_process_coco_resnet34, coco.PostProcessCoco(),
         {"image_size": [1200, 1200, 3]}),
    "coco-1200-onnx":
        (coco.Coco, dataset.pre_process_coco_resnet34, coco.PostProcessCocoOnnx(),
         {"image_size": [1200, 1200, 3]}),
    "coco-1200-pt":
        (coco.Coco, dataset.pre_process_coco_resnet34, coco.PostProcessCocoPt(True,0.05),
         {"image_size": [1200, 1200, 3],"use_label_map": True}),
    "coco-1200-tf":
        (coco.Coco, dataset.pre_process_coco_resnet34, coco.PostProcessCocoTf(),
         {"image_size": [1200, 1200, 3],"use_label_map": False}),
}

# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "imagenet",
        "backend": "tensorflow",
        "cache": 0,
        "max-batchsize": 32,
    },

    # resnet
    "resnet50-tf": {
        "inputs": "input_tensor:0",
        "outputs": "ArgMax:0",
        "dataset": "imagenet",
        "backend": "tensorflow",
        "model-name": "resnet50",
        "in_dtypes": "float32",
    },
    "resnet50-onnxruntime": {
        "dataset": "imagenet",
        "outputs": "ArgMax:0",
        "backend": "onnxruntime",
        "model-name": "resnet50",
    },

    # mobilenet
    "mobilenet-tf": {
        "inputs": "input:0",
        "outputs": "MobilenetV1/Predictions/Reshape_1:0",
        "dataset": "imagenet_mobilenet",
        "backend": "tensorflow",
        "model-name": "mobilenet",
    },
    "mobilenet-onnxruntime": {
        "dataset": "imagenet_mobilenet",
        "outputs": "MobilenetV1/Predictions/Reshape_1:0",
        "backend": "onnxruntime",
        "model-name": "mobilenet",
    },

    # ssd-mobilenet
    "ssd-mobilenet-tf": {
        "inputs": "image_tensor:0",
        "outputs": "num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0",
        "dataset": "coco-300",
        "backend": "tensorflow",
        "model-name": "ssd-mobilenet",
        "in_dtypes": "uint8",
    },
    "ssd-mobilenet-pytorch-native": {
        "inputs": "image",
        "outputs": "bboxes,labels,scores",
        "dataset": "coco-300-pt",
        "backend": "pytorch-native",
        "model-name": "ssd-mobilenet",
    },
    "ssd-mobilenet-pytorch": {
        "dataset": "coco-300",
        "outputs": "num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0",
        "backend": "pytorch",
        "data-format": "NHWC",
        "model-name": "ssd-mobilenet",
    },
    "ssd-mobilenet-onnxruntime": {
        "dataset": "coco-300",
        "outputs": "num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0",
        "backend": "onnxruntime",
        "data-format": "NHWC",
        "model-name": "ssd-mobilenet",
    },

    # ssd-resnet34
    "ssd-resnet34-tf": {
        "inputs": "image:0",
        "outputs": "detection_bboxes:0,detection_classes:0,detection_scores:0",
        "dataset": "coco-1200-tf",
        "backend": "tensorflow",
        "data-format": "NCHW",
        "model-name": "ssd-resnet34",
        "in_dtypes": "float32",
    },
    "ssd-resnet34-pytorch": {
        "inputs": "image",
        "outputs": "bboxes,labels,scores",
        "dataset": "coco-1200-pt",
        "backend": "pytorch-native",
        "model-name": "ssd-resnet34",
    },
    "ssd-resnet34-onnxruntime": {
        "dataset": "coco-1200-onnx",
        "inputs": "image",
        "outputs": "bboxes,labels,scores",
        "backend": "onnxruntime",
        "data-format": "NCHW",
        "max-batchsize": 1,
        "model-name": "ssd-resnet34",
    },
    "ssd-resnet34-onnxruntime-tf": {
        "dataset": "coco-1200-tf",
        "inputs": "image:0",
        "outputs": "detection_bboxes:0,detection_classes:0,detection_scores:0",
        "backend": "onnxruntime",
        "data-format": "NHWC",
        "model-name": "ssd-resnet34",
    },
}


def get_backend(backend):
    if backend == "tensorflow":
        from backend_tf import BackendTensorflow
        backend = BackendTensorflow()
    elif backend == "onnxruntime":
        from backend_onnxruntime import BackendOnnxruntime
        backend = BackendOnnxruntime()
    elif backend == "null":
        from backend_null import BackendNull
        backend = BackendNull()
    elif backend == "pytorch":
        from backend_pytorch import BackendPytorch
        backend = BackendPytorch()
    elif backend == "pytorch-native":
        from backend_pytorch_native import BackendPytorchNative
        backend = BackendPytorchNative()
    elif backend == "tflite":
        from backend_tflite import BackendTflite
        backend = BackendTflite()
    else:
        raise ValueError("unknown backend: " + backend)
    return backend


def get_profile_and_model_path(profile_name: str):
    if profile_name == "resnet50-tf":
        return SUPPORTED_PROFILES[profile_name], "/models/resnet50_v1.pb"
    elif profile_name == "ssd-mobilenet-tf":
        return SUPPORTED_PROFILES[profile_name], "/models/ssd_mobilenet_v1_coco_2018_01_28.pb"
    elif profile_name == "ssd-resnet34-tf":
        return SUPPORTED_PROFILES[profile_name], "/models/resnet34_tf.22.1.pb"
    elif profile_name == "resnet50-onnxruntime":
        return SUPPORTED_PROFILES[profile_name], "/models/resnet50_v1.onnx"
    elif profile_name == "ssd-mobilenet-onnxruntime":
        return SUPPORTED_PROFILES[profile_name], "/models/updated_ssd_mobilenet_v1_coco_2018_01_28.onnx"
    elif profile_name == "ssd-resnet34-onnxruntime":
        return SUPPORTED_PROFILES[profile_name], "/models/updated_resnet34-ssd1200.onnx"
    elif profile_name == "resnet50-pytorch":
        profile = SUPPORTED_PROFILES["resnet50-onnxruntime"]
        profile['backend'] = "pytorch"
        return profile, "/models/resnet50_v1.onnx"
    elif profile_name == "ssd-mobilenet-pytorch":
        return SUPPORTED_PROFILES[profile_name], "/models/updated_ssd_mobilenet_v1_coco_2018_01_28.onnx"
    elif profile_name == "ssd-mobilenet-pytorch-native":
        return SUPPORTED_PROFILES[profile_name], "/models/ssd_mobilenet_v1.pytorch"
    elif profile_name == "ssd-resnet34-pytorch":
        return SUPPORTED_PROFILES[profile_name], "/models/resnet34-ssd1200.pytorch"
    else:
        raise Exception("Unsupported profile: {}".format(profile_name))


def get_img_format(backend_name: str):
    if backend_name == "onnxruntime":
        return "NCHW"
    elif backend_name == "tensorflow":
        return "NHWC"
    elif backend_name == "pytorch":
        return "NCHW"
    else:
        raise Exception("Unsupported backend: {}".format(backend_name))


@dataclass
class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_id, img, label=None):
        self.query_id = query_id
        self.content_id = content_id
        self.img = img
        self.label = label
        self.start = time.time()