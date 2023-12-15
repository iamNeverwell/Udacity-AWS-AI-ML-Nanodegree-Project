checkpoint = 'vgg16_bn_checkpoint.pth'

def load_checkpoint(filepath,device):
    if device=="gpu":
        map_location=lambda device, loc: device.cuda()
    else:
        map_location='cpu'
        
    checkpoint = torch.load(f=filepath,map_location=map_location)

    return checkpoint['model_arch'],checkpoint['clf_input'], checkpoint['clf_output'], checkpoint['clf_hidden'],checkpoint['state_dict'],checkpoint['model_class_to_index']

model_arch,input_units, output_units, hidden_units, state_dict, class_to_idx = load_checkpoint(checkpoint,device)
model.load_state_dict(state_dict)

practice_img = './flowers/test/58/image_02677.jpg'

def process_image(image):
    processed_image = Image.open(image).convert('RGB') 
    processed_image.thumbnail(size=(256,256)) 
    width, height = processed_image.size

    new_width,new_height = 224,224 
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    processed_image = processed_image.crop((left, top, right, bottom))

    transf_tens = transforms.ToTensor()
    transf_norm = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    tensor = transf_norm(transf_tens(processed_image))

    np_processed_image = np.array(tensor)
    return np_processed_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
        plt.tick_params(
            axis='both',          
            which='both',     
            bottom=False,      
            top=False,
            left=False,         
            labelbottom=False,
            labelleft=False,)
        
    image = image.transpose((1, 2, 0))
    

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

from PIL import Image

processed_image = Image.open(practice_img)
processed_image
imshow(process_image(practice_img))

file = 'cat_to_name.json'

def class_to_label(file,classes):
    with open(file, 'r') as f:
        class_mapping =  json.load(f)
        
    labels = []
    for c in classes:
        labels.append(class_mapping[c])
    return labels


idx_mapping = dict(map(reversed, class_to_idx.items()))

def predict(image_path, model,idx_mapping, topk, device):
    pre_processed_image = torch.from_numpy(process_image(image_path))
    pre_processed_image = torch.unsqueeze(pre_processed_image,0).to(device).float()
    
    model.to(device)
    model.eval()
    
    log_ps = model.forward(pre_processed_image)
    ps = torch.exp(log_ps)
    top_ps,top_idx = ps.topk(topk,dim=1)
    list_ps = top_ps.tolist()[0]
    list_idx = top_idx.tolist()[0]
    classes = []
    model.train()
    
    for x in list_idx:
        classes.append(idx_mapping[x])
    return list_ps, classes

def print_predictions(probabilities, classes,image,category_names=None):
    print(image)
    
    if category_names:
        labels = class_to_label(category_names,classes)
        for i,(ps,ls,cs) in enumerate(zip(probabilities,labels,classes),1):
            print(f'{i}) {ps*100:.2f}% {ls.title()} | Class No. {cs}')
    else:
        for i,(ps,cs) in enumerate(zip(probabilities,classes),1):
            print(f'{i}) {ps*100:.2f}% Class No. {cs} ')
    print('') 

probabilities,classes = predict(practice_img,model,idx_mapping,5,device)

print_predictions(probabilities,classes,practice_img.split('/')[-1],file)

imshow(process_image(practice_img))

plt.figure() # image
plt.barh(class_to_label(file,classes),width=probabilities)
plt.title('Model Predictions') 
plt.gca().invert_yaxis()
plt.show()
