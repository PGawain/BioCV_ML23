# BioCV_ML23 - Change Log

## Version 1.4 (Current)

## Enhancements
- Added 2 depth layer in UNet architecture
- Improved weights calculation

### Fixes
- Fixed a bug which caused accuracy weights to be calculated per batch and not on the entire dataset, causing even more spikes than usual

## Version 1.3

### Enhancements
- Removed spir from dataset
- Tested Custom Loss 

### Fixes
- Fixed a bug in preprocessing (causing noisy masks) and retested all previous version to ensure result were consistent

## Version 1.2

### Enhancements
- Added a learning rate scheduler [StepLR(optimizer, step_size=5, gamma=0.1)]
- Added .csv monitoring for learning rate, beta1 and beta2 of Adam optimizer
- Added Dice Loss function
- Added Weighted Dice Loss function implementation to avoid class unbalance
- Considering adding custom loss with weighed dice + croos entropy

## Version 1.1

### Enhancements
- Implemented weight initialization for the neural network (both convolution and up convolution blocks) with He (Kaiming) initialization. 
- Added model checkpointing to save the best model based on validation loss.

### Fixes
- Resolved an issue with attribute error in the `conv_block` class.

###  Results
| Epoch | Train Loss | Val Loss | Val f1 | Accuracy avg | Accuracies                |
|:-----:|:----------:|:--------:|:------:|:-------------:|:-------------------------:|
|  34   |    1.03    |   1.10   |  0.91  |      0.89     | 0.9, 0.81, 0.72, 0.66, 0.79 |




## Version 1.0 (Initial Release)

### Features
- First attempt at a functional UNet.


## To-Do for Future Versions

- [x] change manually the weights distribution in order to increase the accuracy of the model on both kidneys (accuracy_2 and accuracy_3). -> proved not useful
- [x] Add checkpoints
- [x] Verify dead neuron problem or zero gradient
- [x] Remove t2spir from dataset
- [ ] Weight accuracy according to general tendencies and not batch-wise 
- [ ] Verify that train and val sets are balanced case-wise (display one pic each and verify particular cases)

