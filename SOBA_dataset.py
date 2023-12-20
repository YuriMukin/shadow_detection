from torchvision.transforms import Resize
from typing import Optional, Tuple, Callable
from torch.utils.data import Dataset
from helpers import cut_to_tiles
from pandas import DataFrame
from csv import DictReader
from pathlib import Path
from os import listdir
from PIL import Image


class SOBA(Dataset):
    def __init__(self, root: str, split: str, transform: Optional[Callable] = None, image_shape = (120, 160), mode = 'shadow detection'):
        self.mapping(root)

        self.base_folder = Path(root)
        self.csv_file = self.base_folder / ('train.csv' if split =='train' else 'test.csv')
        self.mode = mode
        resizer = Resize(image_shape) #создаем объект для изменения размера изображения
        with open(''+str(self.csv_file)) as csvfile:
           samples = [(''+str(row['Path_image']), ''+str(row['Path_mask']), ''+str(row['Path_shadowless'])) for row in DictReader(csvfile,delimiter=',',skipinitialspace=True)]
        data = []
        for sample in samples:
            image = cut_to_tiles(resizer(Image.open(self.base_folder / sample[0]).convert('RGB'))) #считываем с диска и разделяем на тайлы исходное изображение
            mask = cut_to_tiles(resizer(Image.open(self.base_folder / sample[1]).convert('L'))) #маску тени
            shadowless_image = cut_to_tiles(resizer(Image.open(self.base_folder / sample[2]).convert('RGB'))) #изображение без тени
            for i in range(len(image)):
                if transform is not None: #применяем преобразования (кроп, отражение, отзеркалевание), если они заданны 
                    data.append((transform(image[i]), (transform(mask[i])>0.5).float(), transform(shadowless_image[i])))
                else:
                    data.append((image[i], mask[i], shadowless_image[i]))
        self.data = data
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def mapping(self, directory):
        ttfolds = listdir(directory) #открываем папку с датасетом 
        if not('test.csv' in ttfolds) or not('train.csv' in ttfolds): #проверяем наличие файлов-мапперов
            if 'test.csv' in ttfolds: ttfolds.remove('test.csv') #если какого-то из фалов нет, то удаляем второй
            if 'train.csv' in ttfolds: ttfolds.remove('train.csv')
            map_col = {"test_A" : "Path_image", "test_B" : "Path_mask", "test_C" : "Path_shadowless", "train_A" : "Path_image", "train_B" : "Path_mask", "train_C" : "Path_shadowless"}
            for ttfold in ttfolds: #записываем относительные пути до каждого изображения в соответствующие столбцы 
                data_map = DataFrame(columns=['Path_image', 'Path_mask', 'Path_shadowless'])
                catfolds = listdir(directory + "/" + ttfold)
                for catfold in catfolds:
                    files = listdir(directory + "/" + ttfold + "/" + catfold)
                    files = [ttfold + "/" + catfold + "/" + file for file in files]
                    data_map[map_col[catfold]] = files
                data_map.to_csv(directory + '/' + ttfold + '.csv', index=False)
    
    def __getitem__(self, index: int) -> Tuple:
        sample, mask, shadowless_image = self.data[index]
        if self.mode == 'shadow detection': #в зависимости от типа задачи возвращаем необходимые пары изображений
            return sample, mask
        else:
            sample = sample * mask
            shadowless_image = shadowless_image * mask
            return sample, shadowless_image