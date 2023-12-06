import pydicom as dicom
import os
import cv2
import numpy as np
import skimage.transform
from scipy.ndimage import zoom
import re
from collections import Counter

import matplotlib.pyplot as plt
import math

def zoom_rshp_dicom(files, folder_path, target_dim, target_vox):
    ''' Uses the pydicom library to read the dicom files and then resizes them to the target dimension and voxel dimension. '''
    img_vol = []
    img_dcm_std = None
    for filename in files: 
        img_dcm_std = dicom.dcmread(os.path.join(folder_path,filename))
        img = img_dcm_std.pixel_array
        img_vol.append (img)
    img_vol = np.array(img_vol)

    #Vox dim (x, y, z)
    vox_dim = [img_dcm_std.SpacingBetweenSlices, img_dcm_std.PixelSpacing [0], img_dcm_std.PixelSpacing [1]]
    factor_rshp = [vox_dim[i] / target_vox[i] for i in range(len(target_vox))]
    img_rshp = skimage.transform.rescale(img_vol, factor_rshp, order=3, preserve_range=True)
    
    img_shape = img_rshp.shape
    factor_zoom = [target_dim[i] / img_shape[i] for i in range(len(img_shape))]
    img_zoomed = zoom(img_rshp, factor_zoom, order=3)
    return img_zoomed, vox_dim

def zoom_rshp_ground(files, folder_path, target_dim, target_vox, voxdim):
    ''' Reads the png files and then resizes them to the target dimension and voxel dimension. '''
    img_vol = []
    for filename in files:  
        img = cv2.imread(os.path.join(folder_path,filename), cv2.IMREAD_GRAYSCALE)
        img_vol.append (img)
    img_vol = np.array(img_vol)
    
    factor_rshp = [target_vox[i] / voxdim[i] for i in range(len(target_vox))]
    img_rshp = skimage.transform.rescale(img_vol, factor_rshp, order=0, preserve_range=True)

    img_shape = img_rshp.shape
    factor_zoom = [target_dim[i] / img_shape[i] for i in range(len(img_shape))]
    img_zoomed = zoom(img_rshp, factor_zoom, order=0)

    return img_zoomed 

def decompose(path):
    ''' Utility function that decompose the path into patientID, category and phase. '''
    pattern = r'\d+(?=(?:\\|$))'
    match = re.search(pattern, path)
    patientID = match.group(0) if match else None
    
    category_match = re.search(r'\\(T2SPIR|T1DUAL)(?:\\|$)', path)
    category = category_match.group(1) if category_match else None
    
    phase_match = re.search(r'(InPhase|OutPhase)(?:\\|$)', path)
    phase = phase_match.group(1) if phase_match else None
    
    return patientID, category, phase

def write_log(message):
    try:
        with open("log.txt", 'a') as file:
            file.write(message + '\n')
    except Exception as e:
        print(f"An error occurred while writing to the log file: {e}")

def write_dict(message):
    try:
        with open("result_dict.txt", 'a') as file:
            file.write(message + '\n')
    except Exception as e:
        print(f"An error occurred while writing to the log file: {e}")

def print_result_dict(result_dict):
    for patientID, categories in result_dict.items():
        print(f"Patient ID: {patientID}")
        for category, values in categories.items():
            print(f"  Category: {category}")
            if isinstance(values, dict):
                for phase, data in values.items():
                    print(f"    Phase: {phase} : {data.shape}")
            else:
                for key, value in values.items():
                    print(f"    {key}: {value.shape}")

def write_result_dict(result_dict):
    for patientID, categories in result_dict.items():
        write_dict(f"Patient ID: {patientID}")
        for category, values in categories.items():
            write_dict(f"  Category: {category}")
            if isinstance(values, dict):
                for phase, data in values.items():
                    write_dict(f"    Phase: {phase} : {data.shape}")
            else:
                for key, value in values.items():
                    write_dict(f"    {key}: {value.shape}")

def zoom_rshp_traverse(path, target_dim, target_vox):
    ''' Traverses the directory and calls the zoom_rshp_dicom and zoom_rshp_ground functions. '''
    subdirectories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    result_dict = {}

    for patientID in subdirectories:
        result_dict[patientID] = {}
        subdirectory_path = os.path.join(path, patientID)
        
        # Operate with T1DUAL
        category = "T1DUAL"
        result_dict[patientID][category] = {}
        t1dual_path = os.path.join(subdirectory_path, "T1DUAL", "DICOM_anon")
        ground_t1_path = os.path.join(subdirectory_path, "T1DUAL", "Ground")

        if os.path.exists(ground_t1_path):
            result_dict[patientID][category]["Ground"] = {}
        else:
            write_log(f"Ground folder missing for {patientID}:{category}")
        
        phases = [d for d in os.listdir(t1dual_path) if os.path.isdir(os.path.join(t1dual_path, d))]
        for phase in phases:
            result_dict[patientID][category][phase] = {}
            for root, _, files in os.walk(os.path.join(t1dual_path, phase)):
                if any(file.endswith(".dcm") for file in files):
                    res, voxdim = zoom_rshp_dicom(files, root, target_dim, target_vox)
                    result_dict[patientID][category][phase] = res
                    write_log(f"Adding {root} to the dictionary {patientID}:{category}:{phase}")

        phase = "Ground"
        for root, _, files in os.walk(ground_t1_path):
            if any(file.endswith(".png") for file in files):
                res = zoom_rshp_ground(files, root, target_dim, target_vox, voxdim)
                result_dict[patientID][category][phase] = res

                write_log(f"Adding {root} to the dictionary {patientID}:Ground for {category}")
            else:
                write_log(f"No .png files found in {root}")

        # Operate with T2SPIR
        category = "T2SPIR"
        result_dict[patientID][category] = {}
        t2spir_path = os.path.join(subdirectory_path, "T2SPIR", "DICOM_anon")
        ground_t2_path = os.path.join(subdirectory_path, "T2SPIR", "Ground")
        
        # Check if Ground folder exists
        if os.path.exists(ground_t2_path):
            result_dict[patientID][category]["Ground"] = {}
        else:
            write_log(f"Ground folder missing for {patientID}:{category}")

        phase = "images"
        result_dict[patientID][category][phase] = {}
        for root, _, files in os.walk(t2spir_path):
            if any(file.endswith(".dcm") for file in files):
                res, voxdim = zoom_rshp_dicom(files, root, target_dim, target_vox)
                result_dict[patientID][category][phase] = res
                write_log(f"Adding {root} to the dictionary {patientID}:{category}")

        phase = "Ground"
        for root, _, files in os.walk(ground_t2_path):
            if any(file.endswith(".png") for file in files):
                res = zoom_rshp_ground(files, root, target_dim, target_vox, voxdim)
                result_dict[patientID][category][phase] = res
                write_log(f"Adding {root} to the dictionary {patientID}:Ground for {category}")
            else:
                write_log(f"No .png files found in {root}")

    return result_dict
            
def preprocess():
    ''' Main function that calls the zoom_rshp_traverse function and saves the result dictionary. '''	
    target_dim = [50, 256, 256]
    target_vox = [8.275, 1.65, 1.65]
    result_dict = {}
    #Clear the log file before starting
    open("log.txt", 'w').close()
    open("result_dict.txt", 'w').close()

    result_dict = zoom_rshp_traverse(os.path.join(os.getcwd(), 'Train_Sets', 'MR'), target_dim, target_vox)
    print("Traversing completed.")

    write_dict("RESULT DICTIONARY\nParameters:\n  Target dimension: " + str(target_dim) + "\n  Target voxel dimension: " + str(target_vox) + "\n\n")
    write_result_dict(result_dict)

    np.savez("mr.npz", result_dict)
    print("Dictionary saved.")

def removeSpir():
    ''' Removes the T2SPIR category from the dataset file. '''
    dataset = np.load("mr.npz", allow_pickle=True)
    dataset = dataset['arr_0'].item()
    for patientID, categories in dataset.items():
        for category in list(categories):
            if category == "T2SPIR":
                del dataset[patientID][category]
    
    print_result_dict(dataset)
    np.savez("mr_no_spir.npz", dataset)
