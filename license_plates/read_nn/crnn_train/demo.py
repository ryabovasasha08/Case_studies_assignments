import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
from torchvision import datasets, transforms
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import metrics


def load_images_dict_from_folder(folder):
    chars_dict = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename)).astype(np.float32)
        if img is not None:
            chars_dict[filename] = img
    return chars_dict


import models.crnn as crnn


#loading the pretrained CRNN model
model_path = './data/crnn.pth'
# plate_path = './data/plates'
# dataset = datasets.ImageFolder(plate_path,transform=transformer)
# img_path = './data/demo5.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

#37= 26 alphabets + 10 digits + 1
model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

# +
# data_dir = './data/' 
# transformer = dataset.resizeNormalize((100, 32))
# dataset = datasets.ImageFolder(data_dir, transform=transformer) 

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# +
# # Run this to test your data loader
# images, batch = next(iter(dataloader))
# print(images.size(0))
# print(images[0].shape)
# test = images[0].view(1, *images[0].size())
# print(test.shape)
# test = test.mean(dim=1, keepdim=True)
# print(test.shape)

# img = images[0].permute(1, 2, 0)
# print(img.shape)
# img = np.asarray(img)
# plt.imshow(img)

# +
# plates = []
# for i in range(images.size(0)):
#     plate = images[i]
#     test = plate.view(1, *plate.size())
#     #print(test.shape)
#     test = test.mean(dim=1, keepdim=True)
#     #print(test.shape)
#     plates.append(test)
    
# print(len(plates))
# print(plates[0])
# print(plates[5])
# -

#loading the dataset
plates_dict = load_images_dict_from_folder('./data/plates1')
plates = list(plates_dict.values())
plates_names = list(plates_dict.keys())


# +
#resizing the images to 100x32 as our model expects input of this size
transformer = dataset.resizeNormalize((100, 32))
test_size = len(plates)
print(len(plates))
misident_dict = {}
char_occurence_freq_dict = {}
avg_percent_of_correct_placed_chars = 0
avg_length_identical = 0
avg_text_identical = 0

for p, plate in enumerate(plates):
    text = plates_names[p][0:plates_names[p].find('.')]
    text = text.lower()
    print(text)
    #print(plate.shape)
    temp = Image.fromarray(plate.astype('uint8'))
    plate = temp.convert("L")
    #print(plate.size)
    #plate = torch.tensor(plate)
    plate = transformer(plate)
    #plate = plate.permute(2, 0, 1)
    #print(plate.shape)
    test = plate.view(1, *plate.size())
    #print(test.size())
    
    model.eval()
    preds = model(test)
    
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))
    
    avg_percent_of_correct_placed_chars += metrics.get_correct_placed_chars_percent(text, sim_pred)
    avg_length_identical += metrics.is_length_same(text, sim_pred)
    avg_text_identical +=  metrics.is_text_same(text, sim_pred)
    
    for c_pos, c in enumerate(text):
        if (len(sim_pred)>c_pos and sim_pred[c_pos]!=c):
            if c not in misident_dict:
                misident_dict[c] = [sim_pred[c_pos]]                
            else:
                misident_dict[c].append(sim_pred[c_pos])
        if c not in char_occurence_freq_dict:
            char_occurence_freq_dict[c] = 0
        else:
            char_occurence_freq_dict[c] = char_occurence_freq_dict[c]+1

    #if p == 10:
    #    break
    
# -

avg_percent_of_correct_placed_chars /= test_size
avg_length_identical /= test_size
avg_text_identical /= test_size
print("avg_percent_of_correct_placed_chars: ", avg_percent_of_correct_placed_chars)
print("avg_length_identical: ", avg_length_identical)
print("avg_text_identical: ", avg_text_identical)

misident_nonrepeated_dict = {}
for key in misident_dict:
    misident_nonrepeated_dict[key] = list(dict.fromkeys(misident_dict[key]))

misident_dict

misident_nonrepeated_dict

# +
from operator import itemgetter
import string
x = string.ascii_lowercase+string.digits
misident_freq_list = {}
for char in x:
    if char in misident_dict:
        misident_freq_list[char]=(len(misident_dict[char])/char_occurence_freq_dict[char])*100
    else:
        misident_freq_list[char] = 0
lists = sorted(misident_freq_list.items(),key=lambda x: x[1], reverse=True)
labels, values = zip(*lists) # unpack a list of pairs into two tuples

plt.bar(labels, [value for value in values])
plt.xlabel("characters")
plt.ylabel("Percentage")
plt.title("Frequency of misinterpreting symbols")
# -

# transformer = dataset.resizeNormalize((100, 32)) 
# image = Image.open(img_path).convert('L')
# print(image.size)
# image = transformer(image)
# print(image.size())
# if torch.cuda.is_available():
#     image = image.cuda()
# image = image.view(1, *image.size())
# print(image.size())
# image = Variable(image)
# print(image.size())


# +
#     model.eval()
#     preds = model(image)
#     _, preds = preds.max(2)
#     preds = preds.transpose(1, 0).contiguous().view(-1)
#     preds_size = Variable(torch.IntTensor([preds.size(0)]))
#     raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
#     sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
#     print('%-20s => %-20s' % (raw_pred, sim_pred))

# +
# predict = []
# for plate in plates:
#     model.eval()
#     preds = model(plate)
#     predict.append(preds)

# print(len(predict))

# +
# evaluate = []
# for preds in predict:
#     _, preds = preds.max(2)
#     preds = preds.transpose(1, 0).contiguous().view(-1)
#     preds_size = Variable(torch.IntTensor([preds.size(0)]))
#     raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
#     sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
#     print('%-20s => %-20s' % (raw_pred, sim_pred))
#     evaluate.append(sim_pred)

# #print(evaluate[0].size())
# print(evaluate[1])
# print(len(evaluate))

# +
# #print(evaluate[0].size())
# for e in evaluate:
#     preds_size = Variable(torch.IntTensor([preds.size(0)]))
#     raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
#     sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
#     print('%-20s => %-20s' % (raw_pred, sim_pred))
