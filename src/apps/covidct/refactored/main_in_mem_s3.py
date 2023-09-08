from timeit import default_timer as timer
import os, sys
import numpy as np
import boto3
import io
from skimage.transform import resize
from predict_tf import load_tf_model


s3_client = boto3.client(
    's3',
    aws_access_key_id="YOUR_aws_access_key_id",
    aws_secret_access_key="YOUR_aws_sercret_access_key_id"
)

GPU = True if "USE_GPU" in os.environ and os.environ["USE_GPU"] == "1" else False

def cnn_predict(dataset, cnn_model, cnn_model_outputs, cnn_model_inputs, batchsize):
    length = len(dataset)
    if length < batchsize:
        response = cnn_model.run(cnn_model_outputs, {
            cnn_model_inputs[0]: dataset})
        return response[0].tolist()
    else:
        response = []
        for i in range(0, length, batchsize):
            x = cnn_model.run(cnn_model_outputs, {
                cnn_model_inputs[0]: dataset[i:i+batchsize]})
            response.extend(x[0].tolist())
        return response

def preproc_and_run_first_network(dct_config, plist, bcdu_model, bcdu_inputs, bcdu_outputs, zoom_kernel_dir):
    if GPU:
        import cupy as cp
    from step2 import resample_ct_pixels, truncate_hu, normalize
    patient_type = "T"
    patient_prefix = patient_type + '_'
    num_patient = 0
    dataset = []
    dl_t = 0
    zk_t = 0
    bcdu_t = 0
    other_prep_t = 0
    for name, pinputs in plist.items():
        #patient_id = patient_prefix + patient.strip()
        num_patient += 1
        #print("\n****************************************************************")
        #print("{} / {} : <{}>".format(num_patient,
        #                              str(len(plist)), patient_id))
        #print("****************************************************************")
        # in_npy_ct_pixels_hu = os.path.join(
        #     dct_config['path_ct_pixels_hu_test'], f"{patient_id}_ct-pixels.npy")
        # in_npy_ct_orig_space = os.path.join(
        #     dct_config['path_ct_pixels_hu_test'], f"{patient_id}_ct-spacing.npy")
        # in_npy_ct_orig_shape = os.path.join(
        #     dct_config['path_ct_pixels_hu_test'], f"{patient_id}_ct-orig-shape.npy")

        in_npy_ct_pixels_hu  = pinputs["pixels"]
        in_npy_ct_orig_space = pinputs["spacing"]
        in_npy_ct_orig_shape = pinputs["shape"]

        dl_s = timer()
        try:
            in_npy_ct_pixels_hu.seek(0)
            patient_ct_pixels_hu = np.load(in_npy_ct_pixels_hu, allow_pickle=False)
            
            # this array is too small, leave as numpy
            in_npy_ct_orig_space.seek(0)
            patient_ct_orig_space = np.load(in_npy_ct_orig_space, allow_pickle=False)

            in_npy_ct_orig_shape.seek(0)
            patient_ct_orig_shape = np.load(in_npy_ct_orig_shape, allow_pickle=False)
        except Exception as e:
            print(e)
            sys.exit(1)
        dl_e = timer()

        '''
        Step-01: Resample ct-pixel data
        '''

        patient_ct_resampled_hu, zk = resample_ct_pixels(
            patient_ct_pixels_hu, patient_ct_orig_space, zoom_kernel_dir=zoom_kernel_dir)

            # there is some rounding difference here from CPU to GPU
            # gpu: [ -2753 ; 3758 ]
            # cpu: [ -2758 ; 3765 ]
            # print("ct-resampled_hu HU range: [", np.min(patient_ct_resampled_hu), ";" , np.max(patient_ct_resampled_hu), "]")
        '''
        Step-02: Truncate HU values outside range [-1000;400]
        '''
        if GPU:
            patient_ct_resampled_hu = cp.asnumpy(patient_ct_resampled_hu)
        patient_ct_truncate_hu = truncate_hu(patient_ct_resampled_hu)
        #print("ct-truncate_hu HU range: [", np.min(
        #    patient_ct_truncate_hu), ";", np.max(patient_ct_truncate_hu), "]")
        '''
        Step-04: Normalize
        '''
        patient_ct_norm_hu = normalize(patient_ct_truncate_hu)
        #print(
        #    "ct-norm-hu HU range: [", np.min(patient_ct_norm_hu), ";", np.max(patient_ct_norm_hu), "]")

        CT_resized = resize(
            patient_ct_norm_hu, (patient_ct_norm_hu.shape[0], 128, 128), anti_aliasing=True)

        bcdu_s = timer()
        out = bcdu_model.run(bcdu_outputs, {bcdu_inputs[0]: np.reshape(
            CT_resized, (CT_resized.shape[0], CT_resized.shape[1], CT_resized.shape[2], 1))})
        c = CT_resized-out[0][:, :, :, 0]
        end = timer()
        dataset.append(resize(c, (50, 128, 128)))
        zk_t += zk
        dl_t += dl_e - dl_s
        bcdu_t += end-bcdu_s
        other_prep_t += bcdu_s - dl_e - zk
    dataset = np.array(dataset)
    dataset = np.reshape(
        dataset, (dataset.shape[0], dataset.shape[1], dataset.shape[2], dataset.shape[3], 1))
    return dataset, dl_t, zk_t, bcdu_t, other_prep_t


def main():
    
    modeldl_start = timer()
    #download models into memory
    bcdu_model_iof = io.BytesIO()
    cnn_model_iof = io.BytesIO()
    s3_client.download_fileobj("hf-dgsf", "covid/models/bcdunet_v2.pb", bcdu_model_iof)
    s3_client.download_fileobj("hf-dgsf", "covid/models/cnn_CovidCtNet_v2_final.pb", cnn_model_iof)
    cnn_model_iof.seek(0)
    bcdu_model_iof.seek(0)
    modeldl_end = timer()

    print("downloaded models", file=sys.stderr, flush=True)
    #download inputs

    input_start = timer()
    def download_obj(b, k, io):
        s3_client.download_fileobj(b, k, io)
        io.seek(0)
    p1_pixels_hu = io.BytesIO()
    p1_space = io.BytesIO()
    p1_shape = io.BytesIO()
    p2_pixels_hu = io.BytesIO()
    p2_space = io.BytesIO()
    p2_shape = io.BytesIO()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.submit(download_obj, "hf-dgsf", "covid/inputs/T_normal001_ct-spacing.npy", p1_space    )
        executor.submit(download_obj, "hf-dgsf", "covid/inputs/T_normal001_ct-pixels.npy", p1_pixels_hu )
        executor.submit(download_obj, "hf-dgsf", "covid/inputs/T_normal001_ct-orig-shape.npy", p1_shape )
        executor.submit(download_obj, "hf-dgsf", "covid/inputs/T_normal002_ct-spacing.npy", p2_space   )
        executor.submit(download_obj, "hf-dgsf", "covid/inputs/T_normal002_ct-pixels.npy", p2_pixels_hu)
        executor.submit(download_obj, "hf-dgsf", "covid/inputs/T_normal002_ct-orig-shape.npy", p2_shape)
    input_end = timer()

    zoom_kernel_dir = os.environ["ZOOM_KERNEL_DIR"]
    batchsize = 1

    bcdu_inputs = ["input_1:0"]
    bcdu_outputs = ["conv2d_20/Sigmoid:0"]
    cnn_model_inputs = ["conv3d_1_input:0"]
    cnn_model_outputs = ["dense_2/Softmax:0"]

    plist = {
        "001" : {"pixels" : p1_pixels_hu, "spacing": p1_space, "shape": p1_shape},
        "002" : {"pixels" : p2_pixels_hu, "spacing": p2_space, "shape": p2_shape},
    }

    bcdu_model = load_tf_model(bcdu_model_iof, bcdu_inputs, bcdu_outputs, 0.6)
    bcdu_model.close()
    
    model1_start = timer()
    bcdu_model = load_tf_model(bcdu_model_iof, bcdu_inputs, bcdu_outputs, 0.6)
    dct_config = {'path_ct_pixels_hu_test': './output',
                  'stride': [17, 19, 21],
                  # Define patch sizes for normal patch generation
                  'covid_normal_patch_shape': [3, 128, 128],
                  'hlthy_normal_patch_shape': [3, 128, 128],
                  'pneum_normal_patch_shape': [3, 128, 128],
                  # Define patch sizes for centered annotatted patches
                  'covid_center_patch_shape': [3, 138, 138],
                  'hlthy_center_patch_shape': [3, 128, 128],
                  'pneum_center_patch_shape': [3, 138, 138],
                  'apply_lungs_segmentation': False,
                  'apply_cropping': False}

    model1_end = timer()

    ex1_start = timer()
    dataset, dl_t, zk_t, bcdu_t, other_prep_t = preproc_and_run_first_network(
        dct_config, plist, bcdu_model,
        bcdu_inputs, bcdu_outputs, zoom_kernel_dir)
    ex1_end = timer()
    #bcdu_model.close()

    model2_start = timer()
    cnn_model = load_tf_model(
        cnn_model_iof, cnn_model_inputs, cnn_model_outputs, 0.4)
    model2_end = timer()
    
    ex2_start = timer()
    response = cnn_predict(
        dataset, cnn_model, cnn_model_outputs, cnn_model_inputs, batchsize)
    ex2_end = timer()
    #cnn_model.close()

    # length = len(plist)
    # print(f"end to end: {end - start}")
    # print(f"bcdu model loading: {round(bcdu_ml_e - start, 2)}")
    # print(f"data loading: {round(dl_t, 2)}")
    # print(f"zoom kernel: {round(zk_t, 2)}")
    # print(f"bcdu inf: {round(bcdu_t, 2)}")
    # print(f"other prep: {round(other_prep_t, 2)}")
    # print(f"cnn model loading: {round(cnn_ml_e - cnn_ml_s, 2)}")
    # print(f"cnn inf: {round(end - cnn_ml_e, 2)}")
    # print(f"samples/s: {round(length/(end - start), 1)}")
    # Label = ['Control', 'COVID-19', 'CAP']
    # for i in range(len(response)):
    #     lbl = np.argmax(response[i])
    #     print('The case number %d predicted' %
    #           i, Label[lbl], 'with probibility of %.2f' % (100 * response[i][lbl]))

    model_total = (model1_end-model1_start) + (model2_end-model2_start)
    ex_total = (ex1_end-ex1_start) + (ex2_end-ex2_start)

    ret = {
        "download_input": round((input_end-input_start)*1000, 2),
        "download_model": round((modeldl_end-modeldl_start)*1000, 2),
        "load_model": round(model_total*1000, 2),
        "execution": round(ex_total*1000, 2),
        "end-to-end": round((ex2_end-modeldl_start)*1000, 2)
    }


    import json
    print(">!!"+json.dumps(ret))

    #print("segfaulting to exit..")
    #os.kill(os.getpid(),11)

if __name__ == '__main__':
    main()
