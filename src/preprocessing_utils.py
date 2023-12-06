from collections import Counter
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom


def open_and_check_dim(folder_path):
    """  takes as input a folder path containing DICOM images and returns the shape of the images and the voxel dimensions. It reads the DICOM files in the folder, extracts the pixel arrays, and appends them to a list. It then calculates the voxel dimensions and returns the shape of the image and the voxel dimensions."""
    dcm_ext = '.dcm'
    img_vol = []
    dimensions_list = []
    validFolder = True
    for path, _, files in sorted(os.walk(folder_path)):
        if any(filename.endswith(dcm_ext) for filename in (sorted(files))):
            for filename in (sorted(files)):
                if filename.endswith (dcm_ext):
                    img_dcm_std = dicom.dcmread(os.path.join(folder_path,filename))
                    img = img_dcm_std.pixel_array
                    img_vol.append (img)
                    dimensions_list.append((img.shape, filename))

            z_space = img_dcm_std.SpacingBetweenSlices
            x_space = img_dcm_std.PixelSpacing [0]
            y_space = img_dcm_std.PixelSpacing [1]
            vox_dim_1 = [x_space, y_space, z_space]

            img_vol_raw_1 = np.array (img_vol)
        else:
            validFolder = False
    if validFolder:
        most_common_dimension = Counter(dimensions_list[0]).most_common(1)[0][0]

        stampa = False
        if stampa:
            print("\n" + folder_path)
            print("Dimensione standard più comune:", most_common_dimension)
            for dim, filename in dimensions_list:
                if dim != most_common_dimension:
                    print(f"Immagine: {filename}, Dimensione: {dim}")
            print("Nessuna deviazione in dimensione")

            print ('Original image 1 shape: ', img_vol_raw_1.shape)
            print ('Voxel dimension image 1: ', vox_dim_1)
            print ('\n')
        return img_vol_raw_1.shape, vox_dim_1

def open_and_print_img(folder_path):
    """"This function takes as input the path of the folder containing the images and prints all the images in the folder on a grid."""
    dcm_ext = '.dcm'
    img_vol = []
    dimensions_list = []
    validFolder = True
    img_count = 0
    for path, _, files in sorted(os.walk(folder_path)):
        if any(filename.endswith(dcm_ext) for filename in (sorted(files))):
            for filename in (sorted(files)):
                if filename.endswith (dcm_ext):
                    img_dcm_std = dicom.dcmread(os.path.join(folder_path,filename))
                    img = img_dcm_std.pixel_array
                    img_vol.append (img)
                    img_count += 1

    num_images = len(img_vol)
    # Calcoliamo il numero di righe e colonne per la griglia
    num_cols = 4
    num_rows = math.ceil(num_images / num_cols)

    # Creiamo una griglia di immagini dinamica
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 9))
    fig.suptitle(f'{folder_path}', fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(img_vol[i], cmap='gray')
            ax.set_title(f'Image {i + 1}')
            ax.axis('off')
        else:
            ax.axis('off')    
    plt.tight_layout()
    plt.show()    

def check_dimensions_traverse(path):
    """ Traverse the folders to check dimensions of the images."""
    for root, dirs, files in os.walk(path):
        if any(file.endswith(".dcm") for file in files):
            dimensions, voxels = open_and_check_dim(root)
            if "InPhase" in root:
                in_dim.append(dimensions)
                vox_in_dim.append(voxels)
            elif "OutPhase" in root:
                out_dim.append(dimensions)
                vox_out_dim.append(voxels)
            elif "T2SPIR" in root:
                spir_dim.append(dimensions)
                vox_spir_dim.append(voxels)
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            check_dimensions_traverse(dir_path)

def analyze_dimension(dimensions_list, voxels_list, category):
    """ Analyze the dimensions of the images and prints the results."""
    # Inizializza le liste per le dimensioni
    dim_x = []
    dim_y = []
    dim_z = []
    
    # Inizializza le liste per i voxel
    voxel_x = []
    voxel_y = []
    voxel_z = []
    
    for dim in dimensions_list:
        # Estrai le dimensioni
        dim_x.append(dim[1])  # x
        dim_y.append(dim[2])  # y
        dim_z.append(dim[0])  # z
    
    for voxels in voxels_list:
        # Estrai le dimensioni dei voxel
        voxel_x.append(float(voxels[0]))  # x
        voxel_y.append(float(voxels[1]))  # y
        voxel_z.append(float(voxels[2]))  # z

    # Calcola le medie
    avg_dim_x = sum(dim_x) / len(dim_x)
    avg_dim_y = sum(dim_y) / len(dim_y)
    avg_dim_z = sum(dim_z) / len(dim_z)
    
    avg_voxel_x = sum(voxel_x) / len(voxel_x)
    avg_voxel_y = sum(voxel_y) / len(voxel_y)
    avg_voxel_z = sum(voxel_z) / len(voxel_z)

    # Calcola i massimi
    max_dim_x = max(dim_x)
    max_dim_y = max(dim_y)
    max_dim_z = max(dim_z)

    max_voxel_x = max(voxel_x)
    max_voxel_y = max(voxel_y)
    max_voxel_z = max(voxel_z)

    # Stampa i risultati
    print(f"{category} - Average Dimensions (x, y, z): {avg_dim_x}, {avg_dim_y}, {avg_dim_z}")
    print(f"{category} - Max Dimensions (x, y, z): {max_dim_x}, {max_dim_y}, {max_dim_z}")

    print(f"{category} - Average Voxel dimensions (x, y, z): {avg_voxel_x}, {avg_voxel_y}, {avg_voxel_z}")
    print(f"{category} - Max Voxel dimensions (x, y, z): {max_voxel_x}, {max_voxel_y}, {max_voxel_z}")

def print_directory_tree(root_dir, indent=''):
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            print(indent + '|-- ' + item)
            print_directory_tree(item_path, indent + '|   ')
        else:
            continue
