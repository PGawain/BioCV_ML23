# Description: This file contains all the functions used for training the model
# Import libraries
from sklearn.metrics import confusion_matrix
import torch
import copy
import os
import csv
from tqdm import tqdm
import numpy as np
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torchmetrics.functional.classification import dice as Dice
import pandas as pd

def save_checkpoint(model, optimizer, epoch, best_model_wts) -> None:
    """ Save the model and the optimizer state in a checkpoint file """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_model_wts': best_model_wts
    }
    torch.save(checkpoint, f'training/checkpoint.pt')

def plot_training_info(metrics, csv_path, show_singles=False) -> None:
    """ Plot the training process """
    df = pd.read_csv(csv_path, delimiter=';')
    rows = len(df)
    x_values = range(rows)

    fig, ax = plt.subplots(figsize=(10, 10))

    if show_singles is False:
        for i in range(5):
            metrics.pop(f'accuracy_{i}')


    for key in df.columns:
        if key != 'epoch':
            if key in ['Train_loss', 'Val_loss', 'accuracy'] + ['accuracy_' + str(i) for i in range(5)]:
                if key == 'accuracy':
                    # Highlight the 'accuracy' column
                    ax.plot(x_values, df[key], label=key, linewidth=4)
                else:
                    ax.plot(x_values, df[key], label=key)


    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                        box.width, box.height * 0.8])

    # Put a legend below the current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=True, ncol=5)

    # Display the plot
    plt.show()



def train_model(model, criterion, trainLoader, validationLoader, optimizer, metrics, bpath, num_classes, num_epochs, resume=False, resume_path=None, class_weights=None):    
    """ Train the model. Return the trained model with the best weights and the log of the training process."""
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    current_epoch = 1

    # Resume training from a checkpoint
    if (resume==True and resume_path is not None): 
      checkpoint = torch.load(resume_path)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      current_epoch = checkpoint['epoch'] #sistema
      best_model_wts = checkpoint['best_model_wts'] #sistema

    # Logger
    fieldnames = ['epoch', 'Train_loss', 'Val_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Val_{m}' for m in metrics.keys()] + \
        ['accuracy'] + \
        [f'accuracy_{i}' for i in range(num_classes)]
    
    #adam optimizer logger
    adam_fields = ['epoch', 'lr', 'beta1', 'beta2']
    if not os.path.exists(bpath):
        os.makedirs(bpath)
    file_path = os.path.join(bpath, 'log.csv')
    adam_par_path = os.path.join(bpath, 'adam_params.csv')

    try:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        with open(adam_par_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=adam_fields, delimiter=';')
            writer.writeheader()
    except Exception as e:
        print(f"An error occurred: {e}")

    # Scheduler for learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    
    # Training loop
    for epoch in range(current_epoch, num_epochs+1):
        flag = 0
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
    
        batchsummary = {a: [0] for a in fieldnames}
 
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train() 
                dataloaders=trainLoader
            else:
                model.eval() 
                dataloaders=validationLoader
 
            for sample in tqdm(iter(dataloaders)): 
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                
                optimizer.zero_grad()
 
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)

                    if criterion is not None:
                        loss = criterion(outputs, masks.long())
                    else:
                        #loss = 1 - Dice(outputs, masks.long(), num_classes=num_classes)
                        loss = WeightedDiceLoss(outputs, masks, num_classes=num_classes, weights=class_weights)
                        #loss = CustomLoss(outputs, masks, num_classes=num_classes, device=device)
                        loss = loss.clone().detach().requires_grad_(True)
                    _, outputs = torch.max(outputs, dim=1, keepdim=True)

                    y_pred = outputs.data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()

                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0, average='weighted'))
                        
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                    if phase == 'Val':
                        accuracy, single_accuracy = calculate_accuracy(outputs, masks, num_classes, class_weights)
                        batchsummary['accuracy'].append(accuracy)
                        for i in range(num_classes):
                            batchsummary[f'accuracy_{i}'].append(single_accuracy[i])
                        
                        # Visualizza una immagine di input, la sua ground truth e predizione ogni 10 epoche
                        if flag == 5:
                            # Converte i tensori PyTorch in array NumPy
                            inputs_np = inputs.cpu().squeeze().numpy()
                            #targets_np = masks.cpu().squeeze().numpy()
                            outputs_np = outputs.cpu().squeeze().numpy()

                            #  Visualizza l'immagine di input, la ground truth e la predizione
                            plt.figure(figsize=(12, 4))

                            plt.subplot(1, 3, 1)
                            plt.imshow(inputs_np[0], cmap='gray')
                            plt.title('Input Image')

                            plt.subplot(1, 3, 2)
                            plt.imshow(sample['mask'][0], cmap='gray')
                            plt.title('Ground Truth')

                            plt.subplot(1, 3, 3)
                            plt.imshow(outputs_np[0], cmap='gray')
                            plt.title('Model Prediction')

                            plt.show()
                        flag += 1

            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
        
        scheduler.step()

        for field in fieldnames[3:]:
            values = batchsummary[field]
            batchsummary[field] = np.nanmean(values)

        # Save the lr parameter of the Adam optimizer (for debugging)
        adam_params = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
        }
        try:
            with open(adam_par_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=adam_fields, delimiter=';')
                writer.writerow(adam_params)
        except Exception as e:
            print(f"An error occurred: {e}")

        # Print readable summary
        print_epoch_summary(batchsummary)

        # Save the summary in a csv file
        try:
            with open(file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writerow(batchsummary)
        except Exception as e:
            print(f"An error occurred: {e}")

        # Save the best model
        if phase == 'Val' and epoch_loss < best_loss:
            print('saving model with loss of {}'.format(epoch_loss),
                  'improved over previous {}'.format(best_loss))
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict()) #save the best model weights

        if epoch % 2 == 0:  #Save the best model every 2 epochs to avoid data loss if the training is interrupted
            save_checkpoint(model, optimizer, epoch + 1, best_model_wts)

        if epoch % 5 == 0: # Plot graph depicting the training process improvements every 5 epochs
            plot_training_info(fieldnames, csv_path='training/log.csv', show_singles=True)
 
    # load best model weights 
    model.load_state_dict(best_model_wts)
    return model
    
class ToTensor(object):
    """ Custom function to transform input data to tensor of shape Nc, H, W """
    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        #if img.ndim == 3:
        #    mask = mask.transpose((2, 0, 1))
        #if img.ndim == 3:
        #    img = img.transpose((2, 0, 1))
        return {'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor)}

def print_epoch_summary(batchsummary) -> None:
    print('-' * 40)
    print('{:<15} {:<15} {:<15} {:<15}'.format('Phase', 'Loss', 'WAccuracy', 'F1 Score'))

    for phase in ['Train', 'Val']: 
        print('{:<15} {:<15.4f} {:<15.4f} {:<15.4f}'.format(phase, batchsummary[f'{phase}_loss'], batchsummary['accuracy'], batchsummary[f'{phase}_f1_score'])) 
    print('-' * 40)

def compute_class_weights(y_true, num_classes=5) -> torch.Tensor:
    """ Compute the class weights for any weighted function """
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        weights[i] = torch.sum(y_true == i)
    weights = 1.0 / (weights + 1e-6) # Invert the weights, so that the most frequent class has the lowest weight
    weights = weights / torch.sum(weights) # Normalize the weights, so that they sum to 1
    return weights

def calculate_accuracy(y_pred, y_true, num_classes, class_weights) -> tuple[float, list[float]]:
    """ Calculate the accuracy of the model. The accuracy is computed as the weighted average of the single accuracies """
    if class_weights is None:
        weights = compute_class_weights(y_true, num_classes=num_classes) #return tensor
    else:
        weights = class_weights / torch.sum(class_weights)


    weights = weights.cpu().numpy()
    y_pred = y_pred.data.cpu().numpy().ravel()
    y_true = y_true.data.cpu().numpy().ravel()

    single_accuracy=[]

    for i in range(num_classes):
        correct_pixels  =  np.sum( (y_true == i) & (y_pred == i))
        total_pixels =  np.sum( y_true == i)
        if (total_pixels != 0):
            single_accuracy.append( correct_pixels / total_pixels)
        else:
          single_accuracy.append(np.nan)

    total_accuracy = np.sum(single_accuracy * weights)

    return total_accuracy, single_accuracy

def WeightedDiceLoss(y_pred, y_true, num_classes=5, smooth=1e-6, weights=None) -> torch.Tensor:
    """ Compute the weighted Dice loss """
    # Compute the weights
    if weights is None:
        weights = compute_class_weights(y_true, num_classes)
    else:
        weights = weights / torch.sum(weights)

    # Compute the weighted Dice loss
    loss = 0

    for class_idx in range(0, num_classes):
        y_true_class = (y_true == class_idx).float()        
        y_pred_class = y_pred[:, class_idx, ...]

        intersection = torch.sum(y_true_class * y_pred_class)
        union = torch.sum(y_true_class) + torch.sum(y_pred_class)
        dice = (2.0 * intersection + smooth) / (union + smooth)

        loss += 1.0 - (dice * weights[class_idx])

    return loss

def CustomLoss(y_pred, y_true, num_classes=5, smooth=1e-6, device="cuda:0") -> torch.Tensor:
    """" Custom loss function which is a combination of the weighted Dice loss and the Cross Entropy loss """
    class_weights = compute_class_weights(y_true, num_classes)
    dice_loss = 1 - Dice(y_pred, y_true.long(), num_classes=num_classes)
    total_loss = (0.2 * dice_loss + 0.8 * torch.nn.CrossEntropyLoss(weight=class_weights.to(device))(y_pred, y_true.long()))
    return total_loss

def predict(model, testLoader, num_classes, class_weights=None):
    """ Predict the output of the model on the test set and calculate the accuracy and the confusion matrix"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_true_labels = []
    single_accuracies = []

    with torch.no_grad():
        for sample in tqdm(testLoader):
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)
            
            outputs = model(inputs)
            _, outputs = torch.max(outputs, 1)
            
            y_pred = outputs.data.cpu().numpy().ravel()
            y_true = masks.data.cpu().numpy().ravel()

            all_predictions.extend(y_pred)
            all_true_labels.extend(y_true)

    # Calculate accuracy
    for i in range(num_classes):
        correct_pixels  =  np.sum( (y_true == i) & (y_pred == i))
        total_pixels =  np.sum( y_true == i)
        if (total_pixels != 0):
            single_accuracies.append( correct_pixels / total_pixels)
        else:
          single_accuracies.append(np.nan)
    if class_weights is None:
        single_accuracies = np.array(single_accuracies)
        total_accuracy = np.nanmean(single_accuracies) 
    else:
        single_accuracies = np.array(single_accuracies)
        class_weights = class_weights.cpu().numpy()
        class_weights = class_weights / np.sum(class_weights)
        total_accuracy = np.sum(single_accuracies * class_weights) 


    print(f"Accuracy: {total_accuracy}")
    print('accuracy per class:')
    for i in range(num_classes):
        print(f"Class {i}: {single_accuracies[i]}")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_predictions, labels=range(num_classes))
    conf_df = pd.DataFrame(conf_matrix, index=range(num_classes), columns=range(num_classes))

    print("Confusion Matrix:")
    print(conf_df)

    return total_accuracy, conf_matrix

