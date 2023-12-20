from torch import load, reshape, cat, full, flatten
from torchmetrics.classification import Dice
from torchvision.transforms import ToTensor
from tensorflow.keras.utils import Progbar
import matplotlib.pyplot as plt
from itertools import product
from torch import stack
import numpy as np

PATH = '../models/'


def load_model(model, path, unit):
    model.load_state_dict(load(PATH + path))
    model.eval()
    model.to(unit)
    return model

def smoothHistoryCurve(history, window, key = ''):
    new_history = {}
    if key == '':
        for field_name in history.keys(): 
            field = history[field_name]
            new_history[field_name] = [np.mean(field[i : i + window]) for i in range(0, len(field), window)]
    else:
        new_history[key] = [np.mean(history[key][i : i + window]) for i in range(0, len(history[key]), window)]
    return new_history

def drawGraphs(history, window = -1):
    plt.rcParams['font.size'] = '16'
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    
    if window == -1:
        window = np.int32(len(history['train_loss']) / 1500)

    axs[0].plot(smoothHistoryCurve(history, window, 'train_loss')['train_loss'], label='Train loss')
    axs[0].plot(smoothHistoryCurve(history, window, 'test_loss')['test_loss'], label='Validation loss')
    axs[0].grid(True)
    axs[0].legend(loc=1, prop={'size': 16})

    axs[1].plot(smoothHistoryCurve(history, window, 'train_accuracy')['train_accuracy'], label='DICE on train')
    axs[1].plot(smoothHistoryCurve(history, window, 'test_accuracy')['test_accuracy'], label='DICE on validation')
    axs[1].grid(True)
    axs[1].legend(loc=4, prop={'size': 16})

    plt.show()

def cut_to_tiles(image, tile_size = 32):
    tiles = []
    w, h = image.size
    grid = product(range(0, h-h%tile_size, tile_size), range(0, w-w%tile_size, tile_size))
    for i, j in grid:
        box = (j, i, j+tile_size, i+tile_size)
        tiles.append(np.array(image.crop(box)))
    return tiles

def concatenate_tiles(tiles, image_size = (128, 160, 3), norm = -1):
    tile_size = tiles[0].shape[0]
    image = np.zeros(image_size, float)
    h, w = image_size[:2]
    grid = product(range(0, h-h%tile_size, tile_size), range(0, w-w%tile_size, tile_size))
    count = 0
    for i, j in grid:
        box = (j, j+tile_size, i, i+tile_size)
        if norm == -1:
            image[box[2]:box[3], box[0]:box[1]] = np.array(tiles[count])
        else:
            image[box[2]:box[3], box[0]:box[1]] = np.array(tiles[count]) / norm
        count+=1
    return image

def apply_mask(image, mask):
    res_im = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[0]):
            res_im[i][j][0] *= mask[i][j]
            res_im[i][j][1] *= mask[i][j]
            res_im[i][j][2] *= mask[i][j]
    return res_im

def combine_images(im1, im2, mask):
    image1 = im1.copy()
    image2 = im2.copy()
    for i in range(im1.shape[0]):
        for j in range(im1.shape[0]):
            image1[i][j][0] *= mask[i][j]<0.5
            image1[i][j][1] *= mask[i][j]<0.5
            image1[i][j][2] *= mask[i][j]<0.5
            image2[i][j][0] *= mask[i][j]
            image2[i][j][1] *= mask[i][j]
            image2[i][j][2] *= mask[i][j]
    image = image1 + image2
    return image

def trainNet(unit, model, criterion, optimizer, trainloader, testloader, epochs = 10, semantic_segmentation = False, history = {}):
    if not history:
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': []
        }

    metric = Dice().to(unit)

    for epoch in range(epochs):
        print("epoch {}/{}".format(epoch+1, epochs))
        pbar = Progbar(target=len(trainloader))
        for data in zip(enumerate(trainloader, 0), enumerate(testloader, 0), range(1, len(trainloader) + 1)):
            train, test, idx = data

            train_inputs, train_labels = train[1]
            train_inputs, train_labels = train_inputs.to(unit), train_labels.to(unit)

            test_inputs, test_labels = test[1]
            test_inputs, test_labels = test_inputs.to(unit), test_labels.to(unit)

            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            test_outputs = model(test_inputs)

            if len(train_outputs.size()) != len(train_labels.size()):
                train_outputs = reshape(train_outputs, (train_outputs.size()[0], 1, 32, 32))
                test_outputs = reshape(test_outputs, (test_outputs.size()[0], 1, 32, 32))

            if semantic_segmentation:
                history['train_accuracy'].append(metric(train_outputs, train_labels.long()).item())
                history['test_accuracy'].append(metric(test_outputs, test_labels.long()).item())

                train_outputs = cat(((full(train_outputs.size(), 1).to(unit) - train_outputs).unsqueeze(4), train_outputs.unsqueeze(4)), dim=-1)
                test_outputs = cat(((full(test_outputs.size(), 1).to(unit) - test_outputs).unsqueeze(4), test_outputs.unsqueeze(4)), dim=-1)
                
                train_loss = criterion(reshape(train_outputs, (train_outputs.size()[0]*train_outputs.size()[2]*train_outputs.size()[3], 2)), flatten(train_labels).long())
                test_loss = criterion(reshape(test_outputs, (test_outputs.size()[0]*test_outputs.size()[2]*test_outputs.size()[3], 2)), flatten(test_labels).long())
                history['train_loss'].append(train_loss.item())
                history['test_loss'].append(test_loss.item())
            else:
                train_loss = criterion(train_outputs, train_labels)
                test_loss = criterion(test_outputs, test_labels)
                
            train_loss.backward()
            optimizer.step()

            history['train_loss'].append(train_loss.item())
            history['test_loss'].append(test_loss.item())
            if semantic_segmentation:
                pbar.update(idx, values=[("train_DICE", history["train_accuracy"][-1]), ("train_loss",history["train_loss"][-1]), ("test_DICE", history["test_accuracy"][-1]), ('test_loss', history["test_loss"][-1])])
            else:
                pbar.update(idx, values=[("train_loss",history["train_loss"][-1]), ('test_loss', history["test_loss"][-1])])
        
    return history

def processImage(image, model_resnet_based, crModel, unit):
    image_s_tiles = cut_to_tiles(image)
    image_s_tiles = [ToTensor()(el) for el in image_s_tiles]
    chunks = [image_s_tiles[el:el+(100 if el+100 < len(image_s_tiles) else len(image_s_tiles) - el)] for el in np.arange(0,len(image_s_tiles), 100)]
    mask = np.array([])
    for chunk in chunks:
        image_s_tiles = stack(chunk, dim=0)
        if mask.shape[0] == 0:
            mask = np.squeeze(model_resnet_based(image_s_tiles.to(unit)).detach().cpu().numpy()>0.5)
        else:
            mask = np.concatenate((mask, np.squeeze(model_resnet_based(image_s_tiles.to(unit)).detach().cpu().numpy()>0.5)))
    ful_mask = concatenate_tiles(mask, image_size=(image.size[1], image.size[0]))

    image_tiles = cut_to_tiles(image)
    image_tiles_m = [ToTensor()(apply_mask(image_tiles[i], mask[i])) for i in range(len(image_tiles))]
    chunks = [image_tiles_m[el:el+(100 if el+100 < len(image_tiles_m) else len(image_tiles_m) - el)] for el in np.arange(0,len(image_tiles_m), 100)]
    rest = np.array([])
    for chunk in chunks:
        image_s_tiles = stack(chunk, dim=0)
        if rest.shape[0] == 0:
            rest = np.squeeze(crModel(image_s_tiles.to(unit)).detach().cpu().numpy())
        else:
            rest = np.concatenate((rest, np.squeeze(crModel(image_s_tiles.to(unit)).detach().cpu().numpy())))
    image_tiles = [combine_images(image_tiles[i]/255, ToTensor()(rest[i]).permute(2, 0, 1).numpy(), mask[i]) for i in range(len(image_tiles))]
    image_p = concatenate_tiles(image_tiles, image_size=(image.size[1], image.size[0], 3))
    return ful_mask, image_p