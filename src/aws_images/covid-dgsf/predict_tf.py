#!/usr/bin/env python3
import glob
import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow import dtypes
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from timer import Timer

# GPU = True if "USE_GPU" in os.environ and os.environ["USE_GPU"] == "1" else False
GPU = False

if not GPU:
    from skimage.transform import resize
else:
    import cupy as cp
    cp.cuda.Device(0).use()
    from cucim.skimage.transform import resize


BCDU_MODEL = None
CNN_MODEL = None


def load_tf_model(model_iof, inputs, outputs, mem_pct = 0.45):
    infer_config = tf.compat.v1.ConfigProto()
    infer_config.intra_op_parallelism_threads = int(os.environ['TF_INTRA_OP_PARALLELISM_THREADS']) \
        if 'TF_INTRA_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
    infer_config.inter_op_parallelism_threads = int(os.environ['TF_INTER_OP_PARALLELISM_THREADS']) \
        if 'TF_INTER_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
    
    infer_config.use_per_session_threads = int(os.environ['TF_INTER_OP_PARALLELISM_THREADS'])

    infer_config.gpu_options.allow_growth = True
    infer_config.gpu_options.per_process_gpu_memory_fraction = mem_pct

    graph_def = tf.compat.v1.GraphDef()
    #with tf.compat.v1.gfile.FastGFile(model_path, "rb") as f:
    #    graph_def.ParseFromString(f.read())
    
    graph_def.ParseFromString(model_iof.getvalue())

    as_datatype_enum = dtypes.float32.as_datatype_enum
    optimized_graph_def = optimize_for_inference(graph_def, [item.split(':')[0] for item in inputs],
                                                 [item.split(':')[0] for item in outputs], as_datatype_enum, False)
    graph_def = optimized_graph_def

    g = tf.compat.v1.import_graph_def(graph_def, name='')
    sess = tf.compat.v1.Session(graph=g, config=infer_config)
    return sess


def predict_tf_in_mem(covid_det_input, bcdu_model_path, cnn_model_path, count=None):
    global BCDU_MODEL, CNN_MODEL
    bcdu_inputs = ["input_1:0"]
    bcdu_outputs = ["conv2d_20/Sigmoid:0"]
    with Timer.get_handle("load_bcdu"):
        if BCDU_MODEL is None:
            BCDU_MODEL = load_tf_model(
                bcdu_model_path, bcdu_inputs, bcdu_outputs)

    num_input = len(covid_det_input)

    if not GPU:
        dataset = np.zeros((num_input, 50, 128, 128))
    else:
        dataset = cp.zeros((num_input, 50, 128, 128))
    if count is not None:
        covid_det_input = covid_det_input[:count]
    counter = 0
    for (patient_type, patient_id, CT, depth, height, width) in covid_det_input:
        with Timer.get_handle("resize_128_128_#1"):
            CT_resized = resize(
                CT, (CT.shape[0], 128, 128), anti_aliasing=True)
        with Timer.get_handle("bcdu_predict"):
            if not GPU:
                out = BCDU_MODEL.run(bcdu_outputs, {bcdu_inputs[0]: np.reshape(
                    CT_resized, (CT_resized.shape[0], CT_resized.shape[1], CT_resized.shape[2], 1))})
                c = CT_resized-out[0][:, :, :, 0]
            else:
                reshaped = cp.reshape(
                    CT_resized, (CT_resized.shape[0], CT_resized.shape[1], CT_resized.shape[2], 1))
                out = BCDU_MODEL.run(
                    bcdu_outputs, {bcdu_inputs[0]: reshaped.get()})
                out2 = cp.asarray(out[0])
                c = CT_resized - out2[:, :, :, 0]

        with Timer.get_handle("resize_128_128_#2"):
            dataset[counter] = resize(c, (50, 128, 128))
        counter += 1
    BCDU_MODEL.close()
    del BCDU_MODEL
    gc.collect()

    with Timer.get_handle("reshape"):
        if not GPU:
            dataset = np.reshape(
                dataset, (dataset.shape[0], dataset.shape[1], dataset.shape[2], dataset.shape[3], 1))
        else:
            dataset = cp.reshape(
                dataset, (dataset.shape[0], dataset.shape[1], dataset.shape[2], dataset.shape[3], 1))

    cnn_model_inputs = ["conv3d_1_input:0"]
    cnn_model_outputs = ["dense_2/Softmax:0"]
    with Timer.get_handle("load_cnn"):
        if CNN_MODEL is None:
            CNN_MODEL = load_tf_model(
                cnn_model_path, cnn_model_inputs, cnn_model_outputs)
    with Timer.get_handle("predict"):
        if not GPU:
            x = CNN_MODEL.run(cnn_model_outputs, {
                              cnn_model_inputs[0]: dataset})
        else:
            x = CNN_MODEL.run(cnn_model_outputs, {
                              cnn_model_inputs[0]: dataset.get()})
    CNN_MODEL.close()
    del CNN_MODEL
    gc.collect()

    return x


def predict_tf(input_dir, bcdu_model_path, cnn_model_path, count=None):
    global BCDU_MODEL, CNN_MODEL
    bcdu_inputs = ["input_1:0"]
    bcdu_outputs = ["conv2d_20/Sigmoid:0"]
    with Timer.get_handle("load_bcdu"):
        if BCDU_MODEL is None:
            BCDU_MODEL = load_tf_model(bcdu_model_path,
                                       bcdu_inputs,
                                       bcdu_outputs)

    file_paths = glob.glob(os.path.join(input_dir, '*.npy'))
    if count is not None:
        file_paths = file_paths[:count]

    if not GPU:
        dataset = np.zeros((len(file_paths), 50, 128, 128))
    else:
        dataset = cp.zeros((len(file_paths), 50, 128, 128))

    counter = 0
    for j in file_paths:
        print(f"processing {j}")
        if not GPU:
            CT = np.load(j)
        else:
            CT = cp.load(j)
        with Timer.get_handle("resize_128_128_#1"):
            CT_resized = resize(
                CT, (CT.shape[0], 128, 128), anti_aliasing=True)

        with Timer.get_handle("bcdu_predict"):
            if not GPU:
                out = BCDU_MODEL.run(bcdu_outputs, {bcdu_inputs[0]: np.reshape(
                    CT_resized, (CT_resized.shape[0], CT_resized.shape[1], CT_resized.shape[2], 1))})
                c = CT_resized-out[0][:, :, :, 0]
            else:
                reshaped = cp.reshape(
                    CT_resized, (CT_resized.shape[0], CT_resized.shape[1], CT_resized.shape[2], 1))
                out = BCDU_MODEL.run(
                    bcdu_outputs, {bcdu_inputs[0]: reshaped.get()})
                out2 = cp.asarray(out[0])
                c = CT_resized - out2[:, :, :, 0]

        with Timer.get_handle("resize_128_128_#2"):
            dataset[counter] = resize(c, (50, 128, 128))
        counter += 1

    with Timer.get_handle("reshape"):
        if not GPU:
            dataset = np.reshape(
                dataset, (dataset.shape[0], dataset.shape[1], dataset.shape[2], dataset.shape[3], 1))
        else:
            dataset = cp.reshape(
                dataset, (dataset.shape[0], dataset.shape[1], dataset.shape[2], dataset.shape[3], 1))

    cnn_model_inputs = ["conv3d_1_input:0"]
    cnn_model_outputs = ["dense_2/Softmax:0"]
    with Timer.get_handle("load_cnn"):
        if CNN_MODEL is None:
            CNN_MODEL = load_tf_model(cnn_model_path, cnn_model_inputs,
                                      cnn_model_outputs)

    with Timer.get_handle("predict"):
        if not GPU:
            x = CNN_MODEL.run(cnn_model_outputs, {
                              cnn_model_inputs[0]: dataset})
        else:
            x = CNN_MODEL.run(cnn_model_outputs, {
                              cnn_model_inputs[0]: dataset.get()})
    return x


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, help="limit number of images")
    args = parser.parse_args()
    with Timer.get_handle(f"end-to-end w/ load {'GPU' if GPU else 'CPU'}"):
        input_dir = "./output2"
        bcdu_model = os.path.join("./models", "bcdunet_v2.pb")
        cnn_model = os.path.join("./models", "cnn_CovidCtNet_v2_final.pb")
        x = predict_tf(input_dir, bcdu_model, cnn_model, args.count)
        x = x[0]
        Label = ['Control', 'COVID-19', 'CAP']
        for i in range(x.shape[0]):
            lbl = np.argmax(x[i])
            print('The case number %d is and predicted' % i, Label[lbl], 'with probibility of %.2f' % (100*x[i, lbl]))
    Timer.print()


if __name__ == "__main__":
    main()
