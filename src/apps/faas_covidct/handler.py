from skimage.transform import resize
from timer import Timer
from step1 import get_patient_list
from timeit import default_timer as timer
import os
import sys
import numpy as np
# from step1 import step1_mem
GPU = True if "USE_GPU" in os.environ else False


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


def handle(event, _):
    from predict_tf import load_tf_model
    bcdu_model_path = os.path.join('/models/bcdunet_v2.pb')
    cnn_model_path = os.path.join('/models/cnn_CovidCtNet_v2_final.pb')
    bcdu_inputs = ["input_1:0"]
    bcdu_outputs = ["conv2d_20/Sigmoid:0"]
    cnn_model_inputs = ["conv3d_1_input:0"]
    cnn_model_outputs = ["dense_2/Softmax:0"]

    plist = get_patient_list('/data/patient_list.txt')
    plist = [plist[int(q['idx'])] for q in event['qs']]
    zoom_kernel_dir = '/cuda_dumps'
    batchsize = int(event['batch_size'])

    bcdu_ml_start = timer()
    bcdu_model = load_tf_model(bcdu_model_path, bcdu_inputs, bcdu_outputs)
    bcdu_ml_end = timer()
    dct_config = {'path_ct_pixels_hu_test': '/data',
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

    dataset, dl_t, zk_t, bcdu_t, other_prep_t = preproc_and_run_first_network(
        dct_config, plist, bcdu_model,
        bcdu_inputs, bcdu_outputs, zoom_kernel_dir)

    cnn_ml_start = timer()
    cnn_model = load_tf_model(
        cnn_model_path, cnn_model_inputs, cnn_model_outputs)
    cnn_ml_end = timer()
    response = cnn_predict(dataset, cnn_model, cnn_model_outputs, cnn_model_inputs, batchsize)
    end = timer()
    return {
        'res': response,
        'bcdu_ml': round(bcdu_ml_end-bcdu_ml_start, 2),
        'dl': round(dl_t, 2),
        'zk': round(zk_t, 2),
        'bcdu_inf': round(bcdu_t, 2),
        'other_prep': round(other_prep_t, 2),
        'cnn_ml': round(cnn_ml_end-cnn_ml_start, 2),
        'cnn_inf': round(end-cnn_ml_end, 2),
    }


def main():
    qs = [{'idx': x} for x in range(1)]
    event = {'qs': qs, 'batch_size': 1}
    ret = handle(event, None)
    print(ret['res'])


if __name__ == '__main__':
    main()
