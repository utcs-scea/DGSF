#!/usr/bin/env python3
import os
import sys
import numpy as np
from timer import Timer

#from scipy.ndimage.interpolation import zoom
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
    # new_spacing is always 1,1,1, which is a no-op
    #resize_factor = ct_pixel_spacing / new_spacing
    resize_factor = ct_pixel_spacing

    new_real_shape = ct_pixels.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / ct_pixels.shape
    new_spacing = ct_pixel_spacing / real_resize_factor

    zk_t = 0

    ct_resampled, zk_t = zoom(ct_pixels, real_resize_factor,
                                  mode='nearest', cubin_dir=zoom_kernel_dir)

    #    ct_resampled = zoom(ct_pixels, real_resize_factor, mode='nearest')

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
