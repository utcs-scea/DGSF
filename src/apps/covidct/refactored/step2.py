#!/usr/bin/env python3
import os
import sys
import numpy as np
from timer import Timer

GPU = True if "USE_GPU" in os.environ else False

if not GPU:
    from scipy.ndimage.interpolation import zoom
else:
    import cupy as cp
    # from cupyx.scipy.ndimage import zoom
    from zoom import zoom

MIN_BOUND_HU = -1000.0
MAX_BOUND_HU = 400.0


def build_patient_list(ct_scans_path):
    folders = os.listdir(ct_scans_path)
    patients = []
    for folder in folders:
        if os.path.isdir(os.path.join(ct_scans_path, folder)):  # process only folders
            dcm_files = []
            files = os.listdir(os.path.join(
                ct_scans_path, folder))  # \1 or \RS_3
            for file in files:
                if file.endswith(".dcm"):
                    dcm_files.append(file)
            if len(dcm_files) > 0:
                patients.append(folder)
                print("build_patient_list|Info: patient <{}> found with {} dcm files".format(
                    ct_scans_path + folder, len(dcm_files)))
            else:
                print("build_patient_list|Warn: empty patient data folder ",
                      ct_scans_path + folder)
    print("Patients detected:{}".format(str(len(patients))))
    return patients


# pixel_spacing is a 3 element list.
def resample_ct_pixels(ct_pixels, ct_pixel_spacing, new_spacing=[1, 1, 1],
                       zoom_kernel_dir=os.path.dirname(os.path.realpath(__file__))):
    # TODO: may need to refactor this

    # new_spacing is always 1,1,1, which is a no-op
    #resize_factor = ct_pixel_spacing / new_spacing
    resize_factor = ct_pixel_spacing

    new_real_shape = ct_pixels.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / ct_pixels.shape
    new_spacing = ct_pixel_spacing / real_resize_factor
    # GPU here
    zk_t = 0
    if GPU:
        ct_resampled, zk_t = zoom(ct_pixels, real_resize_factor,
                                  mode='nearest', cubin_dir=zoom_kernel_dir)
    else:
        ct_resampled = zoom(ct_pixels, real_resize_factor, mode='nearest')

    print("resample_ct_pixels|Info ==>",
          "Original shape  :", str(ct_pixels.shape),
          "New shape  :", str(ct_resampled.shape))
    print("resample_ct_pixels|Info ==>",
          "Original spacing:", ct_pixel_spacing,
          "New spacing:", new_spacing)
    return ct_resampled, zk_t


def truncate_hu(ct_img_array):
    # set all hu values outside the range [-1000,400] to -1000 (corresponds to air)
    ct_img_array[ct_img_array > MAX_BOUND_HU] = -1000
    ct_img_array[ct_img_array < MIN_BOUND_HU] = -1000
    return ct_img_array


def normalize(ct_img_array):
    # TODO: we can make this faster with something like https://numpy.org/doc/stable/reference/generated/numpy.apply_over_axes.html
    ct_img_array = (ct_img_array - MIN_BOUND_HU) / \
        (MAX_BOUND_HU - MIN_BOUND_HU)
    ct_img_array[ct_img_array > 1] = 1.
    ct_img_array[ct_img_array < 0] = 0.
    return ct_img_array

# def apply_lung_mask(ct_img_array, lung_mask):
#     ct_lung_seg = ct_img_array.copy()
#     ct_lung_seg[lung_mask == 0] = 0
#     return ct_lung_seg

# not currently used
# def crop_ct_lungs(scan, mask, margin=32):
#     num_slices = mask.shape[0]
#     h_min = mask.shape[1]
#     h_max = 0
#     v_min = mask.shape[2]
#     v_max = 0

#     crop_slices = []

#     for i in range(num_slices):
#         img = mask[i, :, :]
#         img_x_min = max(0, np.min(first_nonzero(img, axis=1, invalid_val=img.shape[1])) - margin)
#         img_x_max = min(img.shape[1], np.max(last_nonzero(img, axis=1, invalid_val=0)) + margin)
#         img_y_min = max(0, np.min(first_nonzero(img, axis=0, invalid_val=img.shape[0])) - margin)
#         img_y_max = min(img.shape[0], np.max(last_nonzero(img, axis=0, invalid_val=0)) + margin)
#         if img_x_min < h_min:
#             h_min = img_x_min
#         if img_x_max > h_max:
#             h_max = img_x_max
#         if img_y_min < v_min:
#             v_min = img_y_min
#         if img_y_max > v_max:
#             v_max = img_y_max

#     for i in range(num_slices):
#         crop_slices.append(scan[i, v_min:v_max, h_min:h_max])
#     scan_crop = np.asarray(crop_slices)
#     print("lung_seg_crop|Info ==> original shape {} --> cropped shape {}".format(scan.shape, scan_crop.shape))
#     return scan_crop


def export_normal_slices(lung_seg_cropped, patch_shape, stride, out_path, patch_npy_prefix, patient_id):
    depth = lung_seg_cropped.shape[0]
    height = lung_seg_cropped.shape[1]
    width = lung_seg_cropped.shape[2]

    if depth > 0:
        fname = patch_npy_prefix + "_" + \
            str(depth).zfill(3) + "_" + str(height).zfill(3) + "_" + str(width).zfill(3) + "_" + \
            patient_id + ".npy"
        out_normal_patch_npy = os.path.join(out_path, fname)

        if not GPU:
            np.save(out_normal_patch_npy, lung_seg_cropped)
        else:
            cp.save(out_normal_patch_npy, lung_seg_cropped)
        print("export_normal_patches|Info: saved patch:", out_normal_patch_npy)
    else:
        print('export_normal_patches|Error: no data to export as patch')


def covid_det_preprocessing_in_mem(dct_config, lst_patients, patient_type, zoom_kernel_dir):
    num_patient = 0
    preprocessed = []
    for patient in lst_patients[:]:
        patient_prefix = patient_type + '_'
        patient_id = patient_prefix + patient.strip()
        num_patient += 1

        print("\n****************************************************************")
        print("{} / {} : <{}>".format(num_patient,
                                      str(len(lst_patients)), patient_id))
        print("****************************************************************")
        in_npy_ct_pixels_hu = os.path.join(
            dct_config['path_ct_pixels_hu_test'], f"{patient_id}_ct-pixels.npy")
        in_npy_ct_orig_space = os.path.join(
            dct_config['path_ct_pixels_hu_test'], f"{patient_id}_ct-spacing.npy")
        in_npy_ct_orig_shape = os.path.join(
            dct_config['path_ct_pixels_hu_test'], f"{patient_id}_ct-orig-shape.npy")

        with Timer.get_handle("load_input"):
            try:
                patient_ct_pixels_hu = np.load(in_npy_ct_pixels_hu)
                # this array is too small, leave as numpy
                patient_ct_orig_space = np.load(in_npy_ct_orig_space)
                patient_ct_orig_shape = np.load(in_npy_ct_orig_shape)

            except Exception as e:
                print(e)
                sys.exit(1)

        '''
        Step-01: Resample ct-pixel data
        '''
        with Timer.get_handle("resample_ct_pixels"):
            patient_ct_resampled_hu, _ = resample_ct_pixels(
                patient_ct_pixels_hu, patient_ct_orig_space, zoom_kernel_dir=zoom_kernel_dir)
            # there is some rounding difference here from CPU to GPU
            # gpu: [ -2753 ; 3758 ]
            # cpu: [ -2758 ; 3765 ]
            # print("ct-resampled_hu HU range: [", np.min(patient_ct_resampled_hu), ";" , np.max(patient_ct_resampled_hu), "]")
        '''
        Step-02: Truncate HU values outside range [-1000;400]
        '''
        with Timer.get_handle("truncate_pixels"):
            if GPU:
                patient_ct_resampled_hu = cp.asnumpy(patient_ct_resampled_hu)
            patient_ct_truncate_hu = truncate_hu(patient_ct_resampled_hu)
            print("ct-truncate_hu HU range: [", np.min(
                patient_ct_truncate_hu), ";", np.max(patient_ct_truncate_hu), "]")

        '''
        Step-04: Normalize
        '''
        with Timer.get_handle("normalize"):
            patient_ct_norm_hu = normalize(patient_ct_truncate_hu)
            print(
                "ct-norm-hu HU range: [", np.min(patient_ct_norm_hu), ";", np.max(patient_ct_norm_hu), "]")

        depth = patient_ct_norm_hu.shape[0]
        height = patient_ct_norm_hu.shape[1]
        width = patient_ct_norm_hu.shape[2]
        preprocessed.append((patient_type, patient.strip(),
                             patient_ct_norm_hu, depth, height, width))
    print("\n*Finished Pre-processing")
    return preprocessed


def covid_det_preprocessing(dct_config, lst_patients, patient_type, train_or_test, save_viz=False):
    normal_patch_shape = [3, 128, 128]
    center_patch_shape = [3, 128, 128]
    num_patient = 0

    for patient in lst_patients[:]:
        patient_prefix = patient_type + '_'
        patient_id = patient_prefix + patient.strip()
        num_patient += 1
        patch_npy_prefix = patient_prefix + str(num_patient).zfill(4)

        print("\n****************************************************************")
        print("{} / {} : <{}>".format(num_patient,
                                      str(len(lst_patients)), patient_id))
        print("****************************************************************")
        in_npy_ct_pixels_hu = os.path.join(
            dct_config['path_ct_pixels_hu_test'], f"{patient_id}_ct-pixels.npy")
        in_npy_ct_orig_space = os.path.join(
            dct_config['path_ct_pixels_hu_test'], f"{patient_id}_ct-spacing.npy")
        in_npy_ct_orig_shape = os.path.join(
            dct_config['path_ct_pixels_hu_test'], f"{patient_id}_ct-orig-shape.npy")
        out_path = dct_config['path_normal_slices_test']

        with Timer.get_handle("load_input"):
            try:
                patient_ct_pixels_hu = np.load(in_npy_ct_pixels_hu)
                # this array is too small, leave as numpy
                patient_ct_orig_space = np.load(in_npy_ct_orig_space)
                patient_ct_orig_shape = np.load(in_npy_ct_orig_shape)

            except Exception as e:
                print(e)
                sys.exit(1)

        '''
        Step-01: Resample ct-pixel data
        '''
        with Timer.get_handle("resample_ct_pixels"):
            patient_ct_resampled_hu, _ = resample_ct_pixels(
                patient_ct_pixels_hu, patient_ct_orig_space)
            # there is some rounding difference here from CPU to GPU
            # gpu: [ -2753 ; 3758 ]
            # cpu: [ -2758 ; 3765 ]
            #print("ct-resampled_hu HU range: [", np.min(patient_ct_resampled_hu), ";" , np.max(patient_ct_resampled_hu), "]")

        '''
        Step-02: Truncate HU values outside range [-1000;400]
        '''
        with Timer.get_handle("truncate_pixels"):
            # back to cpu
            if GPU:
                patient_ct_resampled_hu = cp.asnumpy(patient_ct_resampled_hu)
            patient_ct_truncate_hu = truncate_hu(patient_ct_resampled_hu)
            print("ct-truncate_hu HU range: [", np.min(
                patient_ct_truncate_hu), ";", np.max(patient_ct_truncate_hu), "]")

        # '''
        # Step-03: Compute binary mask for lungs
        # '''
        # if dct_config['apply_lungs_segmentation'] or dct_config['apply_cropping']:
        #     patient_ct_lung_binary_mask = compute_lung_mask(patient_ct_truncate_hu, threshold=-350)

        '''
        Step-04: Normalize
        '''
        with Timer.get_handle("normalize"):
            patient_ct_norm_hu = normalize(patient_ct_truncate_hu)
            print(
                "ct-norm-hu HU range: [", np.min(patient_ct_norm_hu), ";", np.max(patient_ct_norm_hu), "]")

        '''
        Step-05: Apply mask
        '''
        # with Timer.get_handle("segment_lungs"):
        #     if dct_config['apply_lungs_segmentation']:
        #         patient_ct_lung_seg = apply_lung_mask(patient_ct_norm_hu, patient_ct_lung_binary_mask)
        #         print("ct-lung-seg HU range: [", np.min(patient_ct_lung_seg), ";" , np.max(patient_ct_lung_seg), "]")
        #     else:
        #         print("Segmentation of lungs disabled. Set dct_config['apply_lungs_segmentation'] to True to enable")

        '''
        Step-06: Crop Lung Segment
        '''
        # with Timer.get_handle("crop_lung_segments"):
        #     if dct_config['apply_cropping']:
        #         if dct_config['apply_lungs_segmentation']:
        #             patient_ct_lung_seg_cropped = crop_ct_lungs(patient_ct_lung_seg, patient_ct_lung_binary_mask, margin=32)
        #         else:
        #             patient_ct_lung_seg_cropped = crop_ct_lungs(patient_ct_norm_hu, patient_ct_lung_binary_mask, margin=32)
        #     else:
        #         print("Cropping of lungs disabled. Set dct_config['apply_cropping'] to True to enable")

        '''
        Step-07: Export patches(without annotation)
        '''
        with Timer.get_handle("export_patches"):
            # if dct_config['apply_cropping']:
            #     # export_normal_patches(patient_ct_lung_seg_cropped,
            #     #                     normal_patch_shape,
            #     #                     dct_config['stride'],
            #     #                     dct_config['path_normal_patches'],
            #     #                     patch_npy_prefix, patient_id[2:])
            #     export_normal_slices(patient_ct_lung_seg_cropped,
            #                         normal_patch_shape,
            #                         dct_config['stride'],
            #                         out_path,
            #                         patch_npy_prefix, patient_id[2:])
            # else:
            #     # export_normal_patches(patient_ct_norm_hu,
            #     #                     normal_patch_shape,
            #     #                     dct_config['stride'],
            #     #                     dct_config['path_normal_patches'],
            #     #                     patch_npy_prefix, patient_id[2:])
            export_normal_slices(patient_ct_norm_hu,
                                 normal_patch_shape,
                                 dct_config['stride'],
                                 out_path,
                                 patch_npy_prefix, patient_id[2:])
        '''
        # Step-08: Export centered patches(with annotation)
        '''
        # if not annote_csv is None:
        #     pat_id = patient_id[2:]
        #     if pat_id in lst_annot_patients:
        #         df_pat_annot = df_annotation[df_annotation["ID"] == pat_id]
        #         print("Number of annotations:", len(df_pat_annot))
        #         if patient_type=='C' or patient_type=='P':
        #             if dct_config['apply_lungs_segmentation']:
        #                 export_centered_patches(patient_ct_lung_seg,
        #                                         patient_ct_orig_space, patient_ct_orig_shape,
        #                                         df_pat_annot, center_patch_shape,
        #                                         dct_config['path_centered_patches'],
        #                                         patch_npy_prefix, pat_id)
        #             else:
        #                 export_centered_patches(patient_ct_norm_hu,
        #                                         patient_ct_orig_space, patient_ct_orig_shape,
        #                                         df_pat_annot, center_patch_shape,
        #                                         dct_config['path_centered_patches'],
        #                                         patch_npy_prefix, pat_id)
        #         else:
        #             if dct_config['apply_lungs_segmentation']:
        #                 export_random_centered_patches(patient_ct_lung_seg,
        #                                         patient_ct_orig_space, patient_ct_orig_shape,
        #                                         df_pat_annot, center_patch_shape,
        #                                         dct_config['path_centered_patches'],
        #                                         patch_npy_prefix, pat_id)
        #             else:
        #                 export_random_centered_patches(patient_ct_norm_hu,
        #                                         patient_ct_orig_space, patient_ct_orig_shape,
        #                                         df_pat_annot, center_patch_shape,
        #                                         dct_config['path_centered_patches'],
        #                                         patch_npy_prefix, pat_id)

        # if save_viz:
        #     viz_ct_scan(patient_ct_norm_hu, dct_config['path_debug'] + patient_id + '_normalized.pdf')
        #     viz_ct_scan(patient_ct_lung_binary_mask, dct_config['path_debug'] + patient_id + '_mask.pdf')
        #     viz_ct_scan(patient_ct_lung_seg, dct_config['path_debug'] + patient_id + '_lung_seg.pdf')
        #     viz_ct_scan(patient_ct_lung_seg_cropped, dct_config['path_debug'] + patient_id + '_lung_seg_crop.pdf')

    print("\n*Finished Pre-processing")


def step2(dir_ct_scans, input_dir, slices_output_dir, count):
    dct_config = {'path_ct_test': dir_ct_scans,
                  'path_ct_pixels_hu_test': input_dir,
                  'path_normal_slices_test': slices_output_dir,
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

    lst_test_patients = build_patient_list(dct_config['path_ct_test'])
    if count is not None:
        lst_test_patients = lst_test_patients[:count]
    covid_det_preprocessing(dct_config, lst_test_patients,
                            "T", train_or_test="test")


def step2_mem(patient_list, input_dir, zoom_kernel_dir, count=None):
    dct_config = {'path_ct_pixels_hu_test': input_dir,
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
    if count is not None:
        patient_list = patient_list[:count]
    return covid_det_preprocessing_in_mem(dct_config, patient_list, "T", zoom_kernel_dir=zoom_kernel_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, help="limit number of images")
    args = parser.parse_args()
    Timer.reset()
    for _ in range(4):
        with Timer.get_handle(f"e2e step2 {'GPU' if GPU else 'CPU'}"):
            step2("./input/covid-19_cases", "./output", "./output2",
                  args.count)
    Timer.print(ignore_first=1)


if __name__ == "__main__":
    main()
