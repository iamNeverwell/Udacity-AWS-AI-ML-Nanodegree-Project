import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import check_gpu, initial_classifier, validation

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Point to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Point to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    
    return args

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load("checkpoint.pth")
    
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    img = PIL.Image.open(image)

    original_width, original_height = img.size
    
    if original_width < original_height:
        size = [256, 256**600]
    else: 
        size = [256**600, 256]
        
    img.thumbnail(size)
   
    center = original_width/4, original_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    img = img.crop((left, top, right, bottom))

    numpy_img = np.array(img)/255 

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img - mean) / std
        
    numpy_img = numpy_img.transpose(2, 0, 1)
    
    return torch.from_numpy(numpy_img).type(torch.FloatTensor)

def predict(image_path, model, device, cat_to_name, topk=5):
    model.eval()
    
    image_tensor = process_image(image_path)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model.forward(image_tensor)
    
    probabilities, indices = torch.topk(output, topk)
    probabilities = probabilities.exp()

    class_to_idx_inv = {model.class_to_idx[key]: key for key in model.class_to_idx}
    classes = [cat_to_name[class_to_idx_inv[idx]] for idx in indices[0]]
    
    return probabilities[0].tolist(), indices[0].tolist(), classes

def print_probability(probs, flowers):
    for i, j in enumerate(zip(flowers, probs)):
        print("Rank {}:".format(i+1),
              "Flower: {}, likelihood: {}%".format(j[1], ceil(j[0]*100)))

def main():
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    
    device = check_gpu(gpu_arg=args.gpu)
    
    top_probs, top_labels, top_flowers = predict(args.image, model, device, cat_to_name, args.top_k)
    
    print_probability(top_probs, top_flowers)

if __name__ == '__main__':
    main()
