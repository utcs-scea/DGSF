#!/usr/bin/env python3
import os
import pydicom
import numpy as np
from timer import Timer


def load_ct_scan(path):
    dcm_files = []
    for file in os.listdir(path):
        _, file_extension = os.path.splitext(file)
        if file_extension == ".dcm":
            dcm_files.append(os.path.join(path, file))
    try:
        slices = [pydicom.read_file(dcm) for dcm in dcm_files]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        print("load_ct_scan|Info ==> loaded",
              str(len(slices)), "slices from:", path)
        try:
            slice_thickness = np.abs(
                slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(
                slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            if slice_thickness > 0.0:
                s.SliceThickness = slice_thickness
            else:
                print("load_ct_scan|Error: Invalid slice thickness:",
                      slice_thickness)
        spacing = np.array([slices[0].SliceThickness] + list(slices[0].PixelSpacing),
                           dtype=np.float32)
        return slices, spacing
    except Exception as e:
        print("load_ct_scan|Error:", str(e))


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * \
                image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def extract_ct_pixels(lst_patients, input_dir, output_dir, prefix="E", overwrite=True):
    print("*Extracting pixel array data. Input Dir:", input_dir)
    num_patients = 0
    for p in lst_patients[:]:
        num_patients += 1
        print("\n****************************************************************")
        print("{} / {} : <{}>".format(num_patients, str(len(lst_patients)), p))
        print("****************************************************************")

        out_file_ct_pixel = os.path.join(
            output_dir, f"{prefix}{str(p)}_ct-pixels.npy")
        out_file_ct_orig_shape = os.path.join(
            output_dir, f"{prefix}{str(p)}_ct-orig-shape.npy")
        out_file_ct_spacing = os.path.join(
            output_dir, f"{prefix}{str(p)}_ct-spacing.npy")

        if (os.path.isfile(out_file_ct_pixel) == False or os.path.isfile(out_file_ct_orig_shape) == False or os.path.isfile(out_file_ct_spacing) == False or overwrite == True):
            with Timer.get_handle("load_ct_scan"):
                try:
                    patient_ct_slices, patient_ct_spacing = load_ct_scan(
                        os.path.join(input_dir, p))
                    print("Slices(count):", len(patient_ct_slices))
                    #print("Spacing      :", patient_ct_spacing)
                except Exception as e:
                    print("Error! @Function call: load_ct_scan -->", str(e))
                    continue

            with Timer.get_handle("dicom_to_npy"):
                try:
                    patient_ct_pixels = get_pixels_hu(patient_ct_slices)
                    #print("ct-pixels(shape):", patient_ct_pixels.shape)
                except Exception as e:
                    print("Error! @Function call: get_pixels_hu -->", str(e))
                    continue

            with Timer.get_handle("write_output"):
                assert patient_ct_pixels.dtype == 'int16', print("patient_ct_pixels must be of type int16 instead of ",
                                                                 patient_ct_pixels.dtype)
                np.save(out_file_ct_pixel, patient_ct_pixels)
                #print("Saved file: ", out_file_ct_pixel)
                np.save(out_file_ct_orig_shape, patient_ct_pixels.shape)
                #print("Saved file: ", out_file_ct_orig_shape)
                np.save(out_file_ct_spacing, patient_ct_spacing)
                #print("Saved file: ", out_file_ct_spacing)
        else:
            print(
                "Skipped: Output files already exist. Set overwrite = True to force regenerate outputs")


def extract_ct_pixels_in_mem(lst_patients, input_dir, prefix="E"):
    num_patients = 0
    ct_pixels_list = []
    for p in lst_patients[:]:
        num_patients += 1
        with Timer.get_handle("load_ct_scan"):
            try:
                patient_ct_slices, patient_ct_spacing = \
                    load_ct_scan(os.path.join(input_dir, p))
                print("Slices(count):", len(patient_ct_slices))
                # print("Spacing      :", patient_ct_spacing)
            except Exception as e:
                print("Error! @Function call: load_ct_scan -->", str(e))
                continue

        with Timer.get_handle("dicom_to_npy"):
            try:
                patient_ct_pixels = get_pixels_hu(patient_ct_slices)
                # print("ct-pixels(shape):", patient_ct_pixels.shape)
            except Exception as e:
                print("Error! @Function call: get_pixels_hu -->", str(e))
                continue
        assert patient_ct_pixels.dtype == 'int16', \
            print("patient_ct_pixels must be of type int16 instead of ",
                  patient_ct_pixels.dtype)
        ct_pixels_list.append((p, patient_ct_pixels,
                               patient_ct_pixels.shape,
                               patient_ct_spacing))
    return ct_pixels_list


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
                    ct_scans_path + folder,
                    len(dcm_files)))
            else:
                print("build_patient_list|Warn: empty patient data folder ",
                      ct_scans_path + folder)
    print("Patients detected:{}".format(str(len(patients))))
    return patients


def get_patient_list(patient_list):
    patients = []
    with open(patient_list, 'r') as fin:
        for s in fin:
            patient = s.strip()
            patients.append(patient)
    return patients


def step1(dir_ct_scans, output_dir):
    lst_test_patients = build_patient_list(dir_ct_scans)
    # this writes to output_dir/<*.npy>
    extract_ct_pixels(lst_test_patients,
                      dir_ct_scans, output_dir,
                      prefix='T_')


def step1_mem(dir_ct_scans, patient_list):
    lst_test_patients = get_patient_list(patient_list)
    # this writes to output_dir/<*.npy>
    return extract_ct_pixels_in_mem(lst_test_patients,
                                    dir_ct_scans, prefix='T_')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    step1(args.indir, args.outdir)


if __name__ == "__main__":
    main()
