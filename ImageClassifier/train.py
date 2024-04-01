# All code has been referenced from the Udacity Study Material as well as library documentations

#including imports
import os
import argparse
import requests
from pathlib import Path
import tarfile
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
from PIL import Image


def data_transformations(args):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {
        'train_transforms': transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),

        'test_transforms': transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) 
        ]),

        'valid_transforms': transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train_set': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
        'test_set': datasets.ImageFolder(test_dir, transform=data_transforms['test_transforms']),
        'valid_set': datasets.ImageFolder(valid_dir, transform=data_transforms['valid_transforms'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(image_datasets['train_set'], batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(image_datasets['valid_set'], batch_size =64,shuffle = True)
    test_loader = torch.utils.data.DataLoader(image_datasets['test_set'], batch_size = 64, shuffle = True)

    dataloaders = [train_loader, validation_loader, test_loader]
    
    return train_loader, validation_loader, image_datasets['train_set'].class_to_idx

def train_model(args,train_loader, validation_loader, class_to_idx):
    #i'm using resnet since i'm assuming vgg mentioned in the rubrik was an example
    # loading the pretrained network
    
    # resnet18 = models.resnet18(pretrained=True)
    if args.model_arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
    elif args.model_arch == 'resnet34':
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
    else:
        raise ValueError("Unsupported architecture. Please choose between 'resnet18' and 'resnet34'.")
        
    for param in model.parameters():
        param.requires_grad = False
    
    fc = nn.Sequential(
        OrderedDict([
                    ('inputs', nn.Linear(512, 256)), #hidden layer 1 sets output to 120
                    ('relu1', nn.ReLU()),
                    ('dropout',nn.Dropout(0.5)), #could use a different droupout probability,but 0.5 usually works well
                    ('hidden_layer1', nn.Linear(256, 128)), #hidden layer 2 output to 90
                    ('relu2',nn.ReLU()),
                    ('hidden_layer2',nn.Linear(128,102)),#output size = 102
                    ('output', nn.LogSoftmax(dim=1))])
        )

    model.fc = fc
    
    # loss function
    criterion = nn.NLLLoss()
    
    # adam optimizer
    # optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    # optimizer = optim.Adam(resnet18.parameters(), lr=0.001)
    
    # to use the gpu 
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    elif args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print("GPU was selected as the training device, but no GPU is available. Using CPU instead.")
    else:
        device = 'cpu'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f'The device in use is {device}.\n')
    
    # epochs = 20
    epochs = args.epochs
    print_every = 20  # the model trains on 20 batches of images at a time

    running_loss = running_accuracy = 0
    validation_losses, training_losses = [], []

    for e in range(epochs):  # Changed r to e for consistency
        model.train()
        batches = 0

        for images, labels in train_loader:
            batches += 1

            # moves images and labels to the GPU
            images, labels = images.to(device), labels.to(device)

            # forward passes through network
            log_ps = model.forward(images)
            # calculates loss function
            loss = criterion(log_ps, labels)
            # backpropagation
            loss.backward()
            # optimizes the weights
            optimizer.step()

            # calculating metrics
            ps = torch.exp(log_ps)
            top_ps, top_class = ps.topk(1, dim=1)
            matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
            accuracy = matches.mean()

            # resets optimizer 
            optimizer.zero_grad()
            running_loss += loss.item()
            running_accuracy += accuracy.item()

            # runs the model on the validation set every 5 loops
            if batches % print_every == 0:

                # sets the metrics
                validation_loss = 0
                validation_accuracy = 0

                # turns on evaluation mode, turns off calculation of gradients
                model.eval()
                with torch.no_grad():
                    for images, labels in validation_loader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model.forward(images)
                        loss = criterion(log_ps, labels)
                        ps = torch.exp(log_ps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
                        accuracy = matches.mean()

                        # tracks validation metrics
                        validation_loss += loss.item()
                        validation_accuracy += accuracy.item()

                validation_losses.append(validation_loss / len(validation_loader))
                training_losses.append(running_loss / print_every)

                # prints out metrics
                print(f'Epoch {e+1}/{epochs} | Batch {batches}')
                print(f'Running Training Loss: {running_loss / print_every:.3f}')
                print(f'Running Training Accuracy: {running_accuracy / print_every * 100:.2f}%')
                print(f'Validation Loss: {validation_loss / len(validation_loader):.3f}')
                print(f'Validation Accuracy: {validation_accuracy / len(validation_loader) * 100:.2f}%')

                # resets the metrics and turns on training mode
                running_loss = running_accuracy = 0
                model.train()
    
    # saving model checkpoints
    # using fc instead of classifier since that is the last layer in resnet
    model.class_to_idx = class_to_idx
    
    checkpoint = {'fc': model.fc,
                  'arch':args.model_arch,
                  'hidden_layer_1':256,
                  'dropout':0.5,
                  'epochs':args.epochs,
                  'state_dict':model.state_dict(),
                  'optim_stat_dict': optimizer.state_dict(),
                  'class_to_idx':model.class_to_idx
                 }

    torch.save(checkpoint, os.path.join(args.save_directory, "checkpoint.pth"))
    print("model saved to {}".format(os.path.join(args.save_directory, "checkpoint.pth")))
    return True

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # mandatory argument
    # referenced argparse documentation linked in the notebook
    # also referenced the minor project
    parser.add_argument(dest='data_directory', help="This is the dir of the training images")
    
    # optional arguments
    parser.add_argument('--save_directory', dest='save_directory', help="Directory where model will be saved after training.", default='../saved_models')
    parser.add_argument('--learning_rate', dest='learning_rate', help="Learning rate for training the model. Default is 0.003(float)", default=0.001, type=float)
    parser.add_argument('--epochs', dest='epochs', help="Number of epochs for training the model. Default is 5(int)", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', help="For training model on GPU via CUDA", action='store_true')
    # allows to choose between resnet18 and resnet34
    parser.add_argument('--model_arch', dest='model_arch', help="Type of pre-trained model that will be used", default="resnet18", type=str, choices=['resnet18', 'resnet34'])

    # Parse
    args = parser.parse_args()
    
    # Get the directory where train.py is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Combine it with the specified save directory
    save_directory = os.path.join(script_directory, args.save_directory)
    
    # Ensure the save directory exists
    ensure_directory(save_directory)

    # Load and transform data
    train_data_loader, valid_data_loader, class_to_idx = data_transformations(args)

    # Train and save model
    train_model(args,train_data_loader, valid_data_loader, class_to_idx)
