from timeit import default_timer as timer
import os
import numpy as np
from step1 import get_patient_list
from predict_tf import load_tf_model
from handler import preproc_and_run_first_network, cnn_predict

GPU = True if "USE_GPU" in os.environ and os.environ["USE_GPU"] == "1" else False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plist', required=True,
                        help='file contain a list of patient')
    parser.add_argument('--count', type=int, help="limit number of images", default=0)
    parser.add_argument('--batch-size', default=16, type=int)
    args = parser.parse_args()

    count = args.count
    plist = get_patient_list(args.plist)
    if count != 0:
        plist = plist[:count]

    zoom_kernel_dir = '/cuda_dumps'
    batchsize = args.batch_size

    bcdu_model_path = os.path.join('./models/bcdunet_v2.pb')
    cnn_model_path = os.path.join('./models/cnn_CovidCtNet_v2_final.pb')
    bcdu_inputs = ["input_1:0"]
    bcdu_outputs = ["conv2d_20/Sigmoid:0"]
    cnn_model_inputs = ["conv3d_1_input:0"]
    cnn_model_outputs = ["dense_2/Softmax:0"]

    start = timer()
    print("loading bcdu model")
    bcdu_model = load_tf_model(bcdu_model_path, bcdu_inputs, bcdu_outputs, 0.4)
    print("\nloaded\n")
    bcdu_ml_e = timer()
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
    dataset, dl_t, zk_t, bcdu_t, other_prep_t = preproc_and_run_first_network(
        dct_config, plist, bcdu_model,
        bcdu_inputs, bcdu_outputs, zoom_kernel_dir)

    bcdu_model.close()

    cnn_ml_s = timer()
    cnn_model = load_tf_model(
        cnn_model_path, cnn_model_inputs, cnn_model_outputs, 0.4)
    cnn_ml_e = timer()
    response = cnn_predict(
        dataset, cnn_model, cnn_model_outputs, cnn_model_inputs, batchsize)
    end = timer()

    cnn_model.close()

    length = len(plist)
    print(f"end to end: {end - start}")
    print(f"bcdu model loading: {round(bcdu_ml_e - start, 2)}")
    print(f"data loading: {round(dl_t, 2)}")
    print(f"zoom kernel: {round(zk_t, 2)}")
    print(f"bcdu inf: {round(bcdu_t, 2)}")
    print(f"other prep: {round(other_prep_t, 2)}")
    print(f"cnn model loading: {round(cnn_ml_e - cnn_ml_s, 2)}")
    print(f"cnn inf: {round(end - cnn_ml_e, 2)}")
    print(f"samples/s: {round(length/(end - start), 1)}")
    Label = ['Control', 'COVID-19', 'CAP']
    for i in range(len(response)):
        lbl = np.argmax(response[i])
        print('The case number %d predicted' %
              i, Label[lbl], 'with probibility of %.2f' % (100 * response[i][lbl]))


    print("segfaulting to exit..")
    os.kill(os.getpid(),11)

if __name__ == '__main__':
    main()
