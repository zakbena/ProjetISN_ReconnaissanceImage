---
layout: post
title: Fashion MNIST
description: Réseau de neurone Convolutif pour classer FashionMnist
nav-menu: true
---

# Fashion Mnist

Dans cette exemple nous utilisons le jeu de donnée Fashion-Mnist (disponible à l'adresse suivante https://github.com/zalandoresearch/fashion-mnist) pour apprendre a classer des images de vetements.

Cette base de données est composé 60 000 images pour faire l'apprentissage et 10000 images pour tester notre algorithme d'apprentissage, Les images sont réparties en 10 classes qui correspondent à 10 types de vetements différents: dans l'image suivante nous montrons un exemple des 10 types de classes.
![drawing](www/images/fashion_mnist.png"  width="50%" height="50%"/></p>
<p style="text-align:center;"><img src="assets/images/fashion_mnist.png" align="middle"  width="50%" height="50%"/></p>

## Developpement

On utilise la librairie pytorch qui permet de créer des réseaux de neurones de manière optimisé et qui à déjà intégré l'ensemble des formules mathématiques qui permettent de créer un réseau de neurone et de faire son apprentissage.
Nous utilisons la librairie torchvision également afin qu'on puisse envoyer les données dans le réseau de neurone


```python
from chargeur_fashion_mnist import FashionMNIST
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

CUDA = torch.cuda.is_available()
%matplotlib inline
```


```python
# Un objet python pour appliquer des transormation sur les images
transformation = transforms.Compose([
    transforms.ToTensor(), # on va convertir l'image en tenseur ( qui est une matrice à n dimension) de la taille 28x28x1 
    transforms.Normalize((0.1307,), (0.3081,)) # on normalise les couleurs entre 0 et 1
])
# nombre d'image qui sont envoyé en meme temps dans le réseau
taille_batch=64

train_dataset = FashionMNIST('./www/data/fashion_mnist', train=False, download=True, 
                             transform=transformation)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=taille_batch, shuffle=True)

test_dataset = FashionMNIST('./www/data/fashion_mnist', train=False, download=True, 
                            transform=transformation)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=taille_batch, shuffle=True)
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Processing...
    Done!



```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.batchnorm1 = nn.BatchNorm2d(20, affine=True)
        self.conv2 = nn.Conv2d(20, 30, kernel_size=5)
        self.batchnorm2 = nn.BatchNorm2d(30, affine=True)
        
        #self.conv3 = nn.Conv2d(30,10,kernel_size=3)
        #self.batchnorm3 = nn.BatchNorm2d(10, affine=True)

        self.fc1 = nn.Linear(480, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.selu(F.max_pool2d(x,2))
        x = F.selu(F.max_pool2d(self.batchnorm2(self.conv2(x)), 2))
        #x = F.selu(F.max_pool2d(self.batchnorm3(self.conv3(x)), 2))
        x = x.view(x.size(0),-1)
        x = F.selu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```


```python
def apprentissage(epochs=10):
    model = Net()
    if CUDA:
        model = model.cuda()
    model.train()

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adagrad(model.parameters())
    epoch_loss = []
    epoch_accuracy = []
    for epoch in range(epochs):
        batch_loss = []
        for batch_num, (data, targets) in enumerate(train_loader):
            if CUDA:
                data,targets = Variable(data).cuda(), Variable(targets).cuda()
            else:
                data, targets = Variable(data), Variable(targets)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.data[0])
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        accuracy = accuracy_score(targets.data.cpu().numpy(), output.data.cpu().numpy().argmax(axis=1))
        epoch_accuracy.append(accuracy)
        if epoch%5 == 0:
            print('Epoch {}:\t erreur {:.4f}\tprecision {:.2%}'.format(epoch, epoch_loss[-1], accuracy))
        
    return model, epoch_loss,epoch_accuracy

def test_model(model):
    # Test le réseau de neurone sur les données test
    print("Test le réseau sur des données test qui n'on jamais été observé par le réseau")
    model.eval()
    for i,(data, targets) in  enumerate(test_loader):
        if CUDA:
            outputs = model(Variable(data).cuda())
        else:
            outputs = model(Variable(data).cpu())
        log_probs, output_classes = outputs.max(dim=1)
        accuracy = accuracy_score(targets.cpu().numpy(), output_classes.data.cpu().numpy())
        print('Accuracy: {:.2%}'.format(accuracy))
        fig, axes = plt.subplots(8, 8, figsize=(16, 16))

        zip_these = axes.ravel(), log_probs.data.exp(), output_classes.data.cpu(), targets, data.cpu().numpy().squeeze()

        for ax, prob, output_class, target, img in zip(*zip_these):
            ax.imshow(img.reshape(FashionMNIST.input_shape), cmap='gray' if output_class == target else 'autumn')
            ax.axis('off')
            ax.set_title('{} {:.1%}'.format(FashionMNIST.labels[output_class], prob))
        plt.show()
        if i>=5:
            break
  
```


```python
model, epoch_loss,epoch_accuracy = apprentissage(epochs=3)
fig,axes = plt.subplots(1,2, figsize=(16,4))
axes = axes.ravel()
axes[0].plot(epoch_loss)
axes[0].set_xlabel('Epoch')
axes[0].set_title('Erreur')
axes[0].set_ylabel('valeur')

axes[1].plot(epoch_accuracy)
axes[1].set_xlabel('Epoch')
axes[1].set_title('Erreur')
axes[1].set_ylabel('valeur')
plt.show()
test_model(model=model)
```

    /home/zakari/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:23: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number


    Epoch 0:	 erreur 0.7173	precision 75.00%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_6_2.png"  width="50%" height="50%"/></p>


    Test le réseau sur des données test qui n'on jamais été observé par le réseau
    Accuracy: 85.94%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_6_4.png"  width="50%" height="50%"/></p>


    Accuracy: 81.25%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_6_6.png"  width="50%" height="50%"/></p>


    Accuracy: 79.69%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_6_8.png"  width="50%" height="50%"/></p>


    Accuracy: 87.50%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_6_10.png"  width="50%" height="50%"/></p>


    Accuracy: 87.50%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_6_12.png"  width="50%" height="50%"/></p>


    Accuracy: 92.19%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_6_14.png"  width="50%" height="50%"/></p>


    Accuracy: 84.38%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_6_16.png"  width="50%" height="50%"/></p>


    Accuracy: 84.38%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_6_18.png"  width="50%" height="50%"/></p>


    Accuracy: 93.75%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_6_20.png"  width="50%" height="50%"/></p>


    Accuracy: 81.25%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_6_22.png"  width="50%" height="50%"/></p>


    Accuracy: 89.06%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_6_24.png"  width="50%" height="50%"/></p>



```python
model, epoch_loss,epoch_accuracy = apprentissage(epochs=120)
fig,axes = plt.subplots(1,2, figsize=(16,4))
axes = axes.ravel()
axes[0].plot(epoch_loss)
axes[0].set_xlabel('Epoch')
axes[0].set_title('Erreur')
axes[0].set_ylabel('valeur')

axes[1].plot(epoch_accuracy)
axes[1].set_xlabel('Epoch')
axes[1].set_title('Erreur')
axes[1].set_ylabel('valeur')
plt.show()
test_model(model=model)
```

    /home/zakari/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:23: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number


    Epoch 0:	 erreur 0.7239	precision 62.50%
    Epoch 5:	 erreur 0.3574	precision 87.50%
    Epoch 10:	 erreur 0.2847	precision 87.50%
    Epoch 15:	 erreur 0.2230	precision 100.00%
    Epoch 20:	 erreur 0.1847	precision 93.75%
    Epoch 25:	 erreur 0.1523	precision 75.00%
    Epoch 30:	 erreur 0.1246	precision 93.75%
    Epoch 35:	 erreur 0.1065	precision 93.75%
    Epoch 40:	 erreur 0.0818	precision 93.75%
    Epoch 45:	 erreur 0.0701	precision 93.75%
    Epoch 50:	 erreur 0.0584	precision 93.75%
    Epoch 55:	 erreur 0.0471	precision 100.00%
    Epoch 60:	 erreur 0.0396	precision 100.00%
    Epoch 65:	 erreur 0.0345	precision 100.00%
    Epoch 70:	 erreur 0.0276	precision 100.00%
    Epoch 75:	 erreur 0.0264	precision 93.75%
    Epoch 80:	 erreur 0.0224	precision 93.75%
    Epoch 85:	 erreur 0.0194	precision 100.00%
    Epoch 90:	 erreur 0.0173	precision 100.00%
    Epoch 95:	 erreur 0.0157	precision 100.00%
    Epoch 100:	 erreur 0.0135	precision 100.00%
    Epoch 105:	 erreur 0.0125	precision 100.00%
    Epoch 110:	 erreur 0.0115	precision 100.00%
    Epoch 115:	 erreur 0.0099	precision 100.00%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_7_2.png"  width="50%" height="50%"/></p>


    Test le réseau sur des données test qui n'on jamais été observé par le réseau
    Accuracy: 100.00%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_7_4.png"  width="50%" height="50%"/></p>


    Accuracy: 100.00%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_7_6.png"  width="50%" height="50%"/></p>


    Accuracy: 100.00%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_7_8.png"  width="50%" height="50%"/></p>


    Accuracy: 100.00%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_7_10.png"  width="50%" height="50%"/></p>


    Accuracy: 100.00%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_7_12.png"  width="50%" height="50%"/></p>


    Accuracy: 100.00%



<p style="text-align:center;"><img src="assets/images/fashionmnist/output_7_14.png"  width="50%" height="50%"/></p>
