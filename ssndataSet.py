import torch
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from torch.utils.data import Dataset

        
class MyDateSet(Dataset):
    def __init__(self, root_dir, model, transform=None):
        super(MyDateSet, self).__init__()

        self.root_dir = root_dir
        self.model = model
        self.transform = transform

        myAnnotion = pd.read_excel(self.root_dir).values    
        
        self.mean_C = np.around(np.mean(myAnnotion[:, 7]), 2)
        self.std_C = np.around(np.std(myAnnotion[:, 7]), 2)
        self.mean_N = np.around(np.mean(myAnnotion[:, 8]), 2)
        self.std_N = np.around(np.std(myAnnotion[:, 8]), 2)

        ratio = [7, 1, 2] 
        the_index_random = [i for i in range(myAnnotion.shape[0])]
        random.shuffle(the_index_random) 

        train_idx = the_index_random[:int(myAnnotion.shape[0]/sum(ratio)*ratio[0])]
        val_idx = the_index_random[int(myAnnotion.shape[0]/sum(ratio)*ratio[0]):int(myAnnotion.shape[0]/sum(ratio)*(ratio[0]+ratio[1]))]
        test_idx = the_index_random[int(myAnnotion.shape[0]/sum(ratio)*(ratio[0]+ratio[1])):]
        
        if self.model == 'Train':
            myAnnotion = myAnnotion[train_idx]
        elif self.model == 'Val':
            myAnnotion = myAnnotion[val_idx]
        elif self.model == 'Test':
            myAnnotion = myAnnotion[test_idx]
        else:
            raise('不存在' + str(self.model))
        
        self.num_data = myAnnotion.shape[0]

        self.CC = myAnnotion[:, 7]
        self.NC = myAnnotion[:, 8]
        self.label = myAnnotion[:, 12] 
        
        self.data_leaf = myAnnotion[:, 13:217].astype(np.float32)
        self.data_soil = myAnnotion[:, 218:].astype(np.float32)
        scaler = StandardScaler()
        self.data_leaf = scaler.fit_transform(self.data_leaf)
        self.data_soil = scaler.fit_transform(self.data_soil)
            
        import pywt
        self.cA, self.cD = pywt.dwt(self.data_leaf, 'haar')

    def __getitem__(self, index):
        img = torch.FloatTensor(self.data_leaf[index])
        img1 = torch.FloatTensor(self.data_soil[index])
        label = torch.FloatTensor([self.label[index] / 350]) 
        C = torch.FloatTensor([(self.CC[index]-self.mean_C) / self.std_C])
        N = torch.FloatTensor([(self.NC[index]-self.mean_N) / self.std_N])
        cA = torch.FloatTensor(self.cA[index])
        cD = torch.FloatTensor(self.cD[index])
        return img, img1, cA, cD, C, N, label

    def __len__(self):
        return self.num_data
    

           