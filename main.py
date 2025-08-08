import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm
import time
import numpy as np
import pandas as pd
from ssndataSet import MyDateSet
from myTest import plotScatter, plotBar
from ssnModel import Multi_Branch
from metrics import get_regression_metrics


start_time = time.time()

trainModel = True
num_epochs = 200    

modelname = 'wyzmodel.pth'
root_dir = r'E:\SSN\ssn\data\final600.xlsx'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = MyDateSet(root_dir, model="Train", transform=transforms.Compose([transforms.ToTensor()])) 
val_dataset = MyDateSet(root_dir, model="Val", transform=transforms.ToTensor())
test_dataset = MyDateSet(root_dir, model="Test", transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False) 
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

model = Multi_Branch(204, 102, 102, 2, 204, 128, 1)

criterion = nn.MSELoss()

optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': 0.001}], lr=0.001) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1) 

if trainModel:
    lossMin = 2
    for epoch in range(num_epochs):
        model.train()  
        model.to(device)
        lossTatol = 0
        t = tqdm.tqdm(enumerate(train_loader),desc = f'[train]')           
        for step, (img1, img2, img3, img4, img5, img6, label) in t:
            output, _, _, _, _, _ = model(img1.to(device), img2.to(device), img3.to(device), img4.to(device), img5.to(device), img6.to(device))
            loss = criterion(output, label.to(device))
            lossTatol += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
            
        lossAverage = lossTatol/(step+1)
        print('Epoch [{}/{}], AverageLoss: {:.4f}, loss: {:.4f}, lr: {}'.format(epoch+1, num_epochs, lossAverage, loss.item(), optimizer.state_dict()['param_groups'][0]['lr']))

        model.eval()

        lossTatol = 0
        t = tqdm.tqdm(enumerate(val_loader),desc = f'[Test]') 
        for step, (img1, img2, img3, img4, img5, img6, label) in t:
            output = model(img1.to(device), img2.to(device), img3.to(device), img4.to(device), img5.to(device), img6.to(device))
            loss = criterion(output, label.to(device))
            lossTatol += loss.item()

        lossAverage = lossTatol/(step+1)

        if lossMin > lossAverage:
            lossMin = lossAverage
            torch.save(model.state_dict(), modelname)
            print('Model Saved!! lossMin: {:.4f}'.format(lossMin))
        else:
            print('lossMin: {:.4f}, lossNow: {:.4f}'.format(lossMin, lossAverage))
        print(' ')
        
model.load_state_dict(torch.load(modelname))

model.eval()

true_list1 = []
output_list1 = []

lossTatol = 0
t = tqdm.tqdm(enumerate(train_loader),desc = f'[Train]') 
for step, (img1, img2, img3, img4, img5, img6, label) in t:
    output = model(img1.to(device), img2.to(device), img3.to(device), img4.to(device), img5.to(device), img6.to(device))
    loss = criterion(output, label.to(device))
    lossTatol += loss.item()
    true_list1.append(label.cpu().detach().numpy()[0])
    output_list1.append(output.cpu().detach().numpy()[0])
lossAverage = lossTatol/(step+1)

true_list1 = np.array(true_list1)
output_list1 = np.array(output_list1)
print(get_regression_metrics(true_list1*350, output_list1*350))
print('lossNow: {:.10f}'.format(lossAverage))

true_list2 = []
output_list2 = []

lossTatol = 0
t = tqdm.tqdm(enumerate(val_loader),desc = f'[Val]') 
for step, (img1, img2, img3, img4, img5, img6, label) in t:
    output = model(img1.to(device), img2.to(device), img3.to(device), img4.to(device), img5.to(device), img6.to(device))
    loss = criterion(output, label.to(device))
    lossTatol += loss.item()
    true_list2.append(label.cpu().detach().numpy()[0])
    output_list2.append(output.cpu().detach().numpy()[0])
lossAverage = lossTatol/(step+1)

true_list2 = np.array(true_list2)
output_list2 = np.array(output_list2)
print(get_regression_metrics(true_list2*350, output_list2*350))
print('lossNow: {:.10f}'.format(lossAverage))

true_list3 = []
output_list3 = []

lossTatol = 0
t = tqdm.tqdm(enumerate(test_loader),desc = f'[Test]') 
for step, (img1, img2, img3, img4, img5, img6, label) in t:
    output = model(img1.to(device), img2.to(device), img3.to(device), img4.to(device), img5.to(device), img6.to(device))
    loss = criterion(output, label.to(device))
    lossTatol += loss.item()
    true_list3.append(label.cpu().detach().numpy()[0])
    output_list3.append(output.cpu().detach().numpy()[0])

lossAverage = lossTatol/(step+1)

true_list3 = np.array(true_list3)
output_list3 = np.array(output_list3)
result_test = get_regression_metrics(true_list3*350, output_list3*350)
print(result_test)
print('lossNow: {:.10f}'.format(lossAverage))

final_time = time.time() - start_time
print(f'Final Time: {final_time:.2f} seconds')

# plotScatter(true_list3*350, output_list3*350, 'Yield', result_test[0], result_test[1], result_test[2], result_test[3], [0,70,140,210,280,350], [0,350], 250, 75, 25, 25, 25)
# plotBar(true_list3[190:290,0]*350, output_list3[190:290,0]*350, [70,140,210,280,350], 'Yield', [0,20,40,60,80,100]) 
# plotBar(true_list3[:,0]*350, output_list3[:,0]*350, [70,140,210,280,350], 'Yield', [0,80,160,240,320,400,480])   # [0,80,160,240,320,400,480]  [0,10,20,30,40,50,60]

