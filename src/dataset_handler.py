import math
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

# Dataset class loading input images and grounds from dictionary containing the data  
class BioCVDataset(Dataset):
    def __init__(self, dataset, output_classes, transform = None, noSpir=False):
        self.dataset = dataset
        self.transform = transform # If there are transformations to apply
        self.output_classes = output_classes
        self.noSpir = noSpir

    def images_per_patient(self, pedantic=False):
        images_per_patient = 0
        first_patient = list(self.dataset.keys())[0]
        for category in list(self.dataset[first_patient]):
            if pedantic:
                print("Patient " + list(self.dataset.keys())[0]) #First patient used for reference for all the others
                print("Category " + category)
            for phase in list(self.dataset[first_patient][category].keys()):
                if phase != 'Ground':
                    images_per_patient += self.dataset[first_patient][category][phase].shape[0] # number of images per patient (dimension of z axis)
        #assert images_per_patient > 0, "No images found in the dataset"
        #assert images_per_patient == 50*3, "Wrong number of images per patient (default 50 per category)" + str(images_per_patient)
        return images_per_patient

    def __len__(self):
        #number of images in the dataset = number of patients * number of images per category * number of categories
        patients = len(self.dataset.keys()) # number of patients
        return patients * self.images_per_patient() # number of images found in the whole dataset
        
    def __getitem__(self,idx, pedantic=False):
        patient_index = math.floor(idx // self.images_per_patient()) # patient index
        patient_number = sorted(list(self.dataset.keys()))[patient_index] # patient number
        image_number = idx % self.images_per_patient()
        if self.noSpir:
            if image_number < self.images_per_patient() // 2:
                category = 'T1DUAL'
                phase = 'InPhase'
            else:
                category = 'T1DUAL'
                phase = 'OutPhase'
                image_number -= self.images_per_patient() // 2
        else:
            if image_number < self.images_per_patient() // 3: #first third, T1DUAL, InPhase
                category = 'T1DUAL'
                phase = 'InPhase'
            elif image_number < 2 * self.images_per_patient() // 3: #second third, T1DUAL, OutPhase
                category = 'T1DUAL'
                phase = 'OutPhase'
                image_number -= self.images_per_patient() // 3
            else: #third third, T2SPIR
                category = 'T2SPIR'
                phase = 'images'
                image_number -= 2 * self.images_per_patient() // 3

        if pedantic:
            print(f"Index: {idx}")
            print(f"Patient number: {patient_number}")
            print(f"Category: {category}")
            print(f"Phase: {phase}")
            print(f"Image number: {image_number}")

        img = self.dataset[patient_number][category][phase][image_number]
        mask = self.dataset[patient_number][category]['Ground'][image_number]

        mask[(mask >= 55) & (mask <= 70)] = 1 # Liver
        mask[(mask >= 110) & (mask <= 135)] = 2 # Right kidney
        mask[(mask >= 175) & (mask <= 200)] = 3 # Left kidney
        mask[(mask >= 240) & (mask <= 255)] = 4 # Spleen
        #If it is not a pixel of interest, it is background
        mask[(mask != 1) & (mask != 2) & (mask != 3) & (mask != 4)] = 0

        sample = {'image': np.expand_dims(img, axis=0), 'mask': mask} # matched image and mask
    
        
        if self.transform:
            sample=self.transform(sample) # Eventual transformation to be made on the input data
        mask = sample['mask']
        
        return sample
    
def train_val_split(dataset, validation_size=0.15, pedantic=False, noSpir=False):
    """ Split the dataset in training and validation set. Return the two sets."""
    train_dataset = {}
    val_dataset = {}
    test_dataset = {}
    patients_num = len(dataset.keys())
    if pedantic:
        print("-" * 80)
        print(f"Total number of patients: {patients_num}")
    train_patients_num = math.floor((1 - validation_size) * patients_num)
    val_patients_num = math.floor((patients_num - train_patients_num)/2)
    if pedantic:
        print(f"Expected number of patients in the training set: {train_patients_num}")
        print(f"Expected number of patients in the validation set: {patients_num - train_patients_num}")
    training_patients = 0
    val_patiens = 0
    for patient in list(dataset.keys()):
        if training_patients < train_patients_num:
            train_dataset[patient] = {}
            for category in list(dataset[patient].keys()):
                train_dataset[patient][category] = {}
                for phase in list(dataset[patient][category].keys()):
                    data = dataset[patient][category].get(phase)
                    train_dataset[patient][category][phase] = data
            training_patients += 1
        else:
            if val_patiens < val_patients_num:
                val_dataset[patient] = {}
                for category in list(dataset[patient].keys()):
                    val_dataset[patient][category] = {}
                    for phase in list(dataset[patient][category].keys()):
                        data = dataset[patient][category].get(phase)
                        val_dataset[patient][category][phase] = data
                val_patiens += 1
            else: #test set
                test_dataset[patient] = {}
                for category in list(dataset[patient].keys()):
                    test_dataset[patient][category] = {}
                    for phase in list(dataset[patient][category].keys()):
                        data = dataset[patient][category].get(phase)
                        test_dataset[patient][category][phase] = data

    if pedantic:
        print(f"Number of patients in the training set: {len(train_dataset.keys())}")
        print(f"Number of patients in the validation set: {len(val_dataset.keys())}")
        #List patients in the both sets
        print(f"Patients in the training set: {list(train_dataset.keys())}")
        print(f"Patients in the validation set: {list(val_dataset.keys())}")   
        print("-" * 80)
    return train_dataset, val_dataset, test_dataset