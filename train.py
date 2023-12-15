import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

data_dir = 'flowers'
print(data_dir)
train_dir = data_dir + '/train'
print(train_dir)
valid_dir = data_dir + '/valid'
print(valid_dir)
test_dir = data_dir + '/test'
print(test_dir)

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
print(train_data)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
print(test_data)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
print(valid_data)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

image_datasets = [train_data, valid_data, test_data]
dataloaders = [trainloader, validloader, testloader]

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    print(cat_to_name)

no_output_categories = len(cat_to_name)

hidden_units = 4096
model = models.vgg16_bn(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.classifier

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),
                          ('fc2', nn.Linear(hidden_units, no_output_categories)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

model.classifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f'The device in use is {device}.\n')


epochs = 10
optimizer = optim.Adam(model.classifier.parameters(),lr=.001)
criterion = nn.NLLLoss()

print_every = 20

running_loss = running_accuracy = 0
validation_losses, training_losses = [],[]


for e in range(epochs):
    batches = 0

    model.train()

    for images,labels in trainloader:
        start = time.time() 
        batches += 1

        images,labels = images.to(device),labels.to(device)

        log_ps = model.forward(images)
        loss = criterion(log_ps,labels)
        loss.backward()
        optimizer.step()

        ps = torch.exp(log_ps)
        top_ps, top_class = ps.topk(1,dim=1)
        matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
        accuracy = matches.mean()

        optimizer.zero_grad()
        running_loss += loss.item()
        running_accuracy += accuracy.item()

        if batches%print_every == 0:
            end = time.time()
            training_time = end-start
            start = time.time()

            validation_loss = 0
            validation_accuracy = 0

            model.eval()
            with torch.no_grad():
                for images,labels in validloader:
                    images,labels = images.to(device),labels.to(device)
                    log_ps = model.forward(images)
                    loss = criterion(log_ps,labels)
                    ps = torch.exp(log_ps)
                    top_ps, top_class = ps.topk(1,dim=1)
                    matches = (top_class == \
                                labels.view(*top_class.shape)).type(torch.FloatTensor)
                    accuracy = matches.mean()

                    validation_loss += loss.item()
                    validation_accuracy += accuracy.item()
      
            end = time.time()
            validation_time = end-start
            validation_losses.append(running_loss/print_every)
            training_losses.append(validation_loss/len(validloader))
                
            print(f'Epoch {e+1}/{epochs} | Batch {batches}')
            print(f'Running Training Loss: {running_loss/print_every:.3f}')
            print(f'Running Training Accuracy: {running_accuracy/print_every*100:.2f}%')
            print(f'Validation Loss: {validation_loss/len(validloader):.3f}')
            print(f'Validation Accuracy: {validation_accuracy/len(validloader)*100:.2f}%')

            running_loss = running_accuracy = 0
            model.train()

test_accuracy = 0
start_time = time.time()
print('Validation started.')
for images,labels in testloader:
    model.eval()
    images,labels = images.to(device),labels.to(device)
    log_ps = model.forward(images)
    ps = torch.exp(log_ps)
    top_ps,top_class = ps.topk(1,dim=1)
    matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
    accuracy = matches.mean()
    test_accuracy += accuracy

nd_time = time.time()
print('Validation ended.')
validation_time = end_time - start_time
print('Validation time: {:.0f}m {:.0f}s'.format(validation_time / 60, validation_time % 60))

print(f'Test Accuracy: {test_accuracy/len(testloader)*100:.2f}%')

destination_directory = None
class_to_idx = train_data.class_to_idx

def save_model(trained_model,hidden_units,output_units,destination_directory,model_arch,class_to_idx):
   
    model_checkpoint = {'model_arch':model_arch, 
                    'clf_input':25088,
                    'clf_output':output_units,
                    'clf_hidden':hidden_units,
                    'state_dict':trained_model.state_dict(),
                    'model_class_to_index':class_to_idx,
                    }
    
    if destination_directory:
        torch.save(model_checkpoint,destination_directory+"/"+model_arch+"_checkpoint.pth")
        print(f"{model_arch} successfully saved to {destination_directory}")
    else:
        torch.save(model_checkpoint,model_arch+"_checkpoint.pth")
        print(f"{model_arch} successfully saved to current directory as {model_arch}_checkpoint.pth")

save_model(model,hidden_units,no_output_categories,destination_directory,'vgg16_bn',class_to_idx)

