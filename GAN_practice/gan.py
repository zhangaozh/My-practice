import torch
import torch.nn as nn
from torch.nn import ReLU, Module
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import model
from config import configurations

    
def switch_img(x):
    out = 0.5 * (x + 1) 
    out = out.view(-1, 1, 28, 28)
    return out    

if __name__ == '__main__':

    # ======= hyperparameters & data loaders =======#
    cfg = configurations[1]
    MODEL_ROOT = cfg['MODEL_ROOT']  # the root to buffer your checkpoints
    Generator_root = cfg['Generator_root']   # the root to buffer your generated image
    DEVICE = cfg['DEVICE']
    NUM_EPOCH = cfg['NUM_EPOCH']
    BATCH_SIZE = cfg['BATCH_SIZE']
    LR = cfg['LR']
    Z_DIMENSION = cfg['Z_DIMENSION']
    

    #prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    # test_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

    G = model.Generator()
    G = G.to(DEVICE)
    D = model.Discriminator()
    D = D.to(DEVICE)
    
    LOSS = nn.BCELoss()
    # LOSS = nn.CrossEntropyLoss()

    D_optimizer = torch.optim.Adam(D.parameters(), lr=LR)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=LR)

    for epoch in range(NUM_EPOCH):
        
        D.train()
        G.train()

        all_D_loss = 0
        all_G_loss = 0
        num_step = 0

        for steps, (image, label) in enumerate(train_loader):
    
            image = image.to(DEVICE) 
            label = label.to(DEVICE)

            num_img = label.size(0)

            R_labels = torch.ones_like(label, dtype=torch.float)
            F_labels = torch.zeros_like(label, dtype=torch.float)
        
            #train Discrimator
            for _ in range(4):
            
                #train R_data
                # R_data = torch.faltten(image, strat_dim=1)
                R_data = image.view([image.size()[0], -1])
                R_outputs = D(R_data)
                R_loss_D = LOSS(R_outputs, R_labels)
            
                z = torch.randn((num_img, Z_DIMENSION))
                z = z.to(DEVICE)  # Random noise from N(0,1)
            
                F_data = G(z)  # Generate fake data
                F_outputs = D(F_data.detach())  #stop gradient
                F_loss_D = LOSS(F_outputs, F_labels)

                Loss_D = R_loss_D + F_loss_D

                D_optimizer.zero_grad()  #zero or 累加进下面的D梯度
                Loss_D.backward()
                D_optimizer.step()

            # Train Generator
            z = torch.randn((num_img, Z_DIMENSION))
            z = z.to(DEVICE)

            F_data = G(z)
        
            outputs_G = D(F_data)
            Loss_G = LOSS(outputs_G, R_labels)

            G_optimizer.zero_grad()
            Loss_G.backward()
            G_optimizer.step()

            all_D_loss += Loss_D.item()
            all_G_loss += Loss_G.item()
            num_step = steps + 1
        print('Epoch: {}/{}, D_loss: {:.6f}, G_loss: {:.6f} '.format
              (epoch, NUM_EPOCH, all_D_loss/(num_step), all_G_loss/(num_step)))
        print("=" * 60)

        # Save generated images
        Generated_images = switch_img(F_data)
        save_image(Generated_images, os.path.join(Generator_root, 'Generated_images-{}.png'.format(epoch + 1)))

        torch.save(G.state_dict(), os.path.join(MODEL_ROOT,
                                        "Generator_Epoch_{}_checkpoint.pth".format(epoch + 1)))
        torch.save(D.state_dict(), os.path.join(MODEL_ROOT,
                                        "Discriminator_Epoch_{}_checkpoint.pth".format(epoch + 1)))
