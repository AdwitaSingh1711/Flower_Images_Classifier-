# All code has been referenced from the Udacity Study Material as well as library documentations

#including imports
import argparse
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import json


def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    # model = models.resnet18(pretrained=True)

    # choose model arch as per checkpoint(according to rubric)
    if checkpoint['arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif checkpoint['arch'] == 'resnet34':
        model = models.resnet34(pretrained=True)
    
    #to freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    #load from checkpoint
    # resnet18.class_to_idx = checkpoint['class_to_idx']
    # resnet18.load_state_dict(checkpoint['state_dict'])
    # Map keys from checkpoint to match current model architecture
    
    # changed resnet18 var to model 
    # to generalise for model arch according to checkpoint
    model.fc = checkpoint['fc']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #size = 256, 256
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    image = img_transforms(img_pil)
    
    return image


''' didn't end up needing this as per rubric so commenting it out

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax'''

def predict(image_path, model, device, cat_to_name,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # arguement errors when using device that was passed in the function
    # declaring this again here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    img = process_image(image_path)
    img = img.numpy()
    # img = torch.from_numpy(np.array([img])).float()
    img = torch.from_numpy(np.array([img])).float().to(device)

    with torch.no_grad():
        # output = resnet18.forward(img.cuda())
        output = model.forward(img)
        
    probability = torch.exp(output).data
    ps, top_classes = probability.topk(topk, dim=1)
    
    idx_to_flower = {}
    for i, j in model.class_to_idx.items():
        idx_to_flower[j] = cat_to_name[i]
    
    # map the predicted classes
    list_of_predicted_flowers = []
    for i in top_classes.tolist()[0]:
        list_of_predicted_flowers.append(idx_to_flower[i])
    
    return ps.tolist()[0], list_of_predicted_flowers

def print_probabilities(args,device):
    # load model
    model = load_checkpoint(args.model_filepath)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # resnet18 = resnet18.to(device)
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    elif args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print("GPU was selected as the training device, but no GPU is available. Using CPU instead.")
    else:
        device = 'cpu'
    model.to(device)
    print(f'The device in use is {device}.\n')

    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)

    # predicting image
    top_ps, top_classes = predict(args.image_filepath, model, args.gpu, cat_to_name,args.top_k)

    print("Predicted Class Probabilities for the flower:")
    for i in range(args.top_k):
        print(f"#{i:<3} {top_classes[i]:<25} Probability: {top_ps[i]*100:.2f}%")
            
if __name__ == '__main__':
    
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(dest='image_filepath', help="This is the image file that you want to classify")
    # />home>workspace>ImageClassifier>flowers>test>102>image_08004.jpg
    parser.add_argument(dest='model_filepath', help="This is file path of a checkpoint file, including the extension")
    # />home>workspace>saved_models>checkpoint.pth

    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath', help="File path to the json file that maps categories to real names", default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', help="Number of most likely classes to return, default is 5(int)", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', help="For training model on GPU via CUDA(string)", action='store_true')

    # Parse and print the results
    args = parser.parse_args()
    
    # Check if GPU is requested and available
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU for prediction")
    else:
        device = torch.device("cpu")
        print("Using CPU for prediction")
        
    # model = load_checkpoint(args.model_filepath)

    print_probabilities(args,device)
    