---
layout: post
title: MNIST (amélioré)
description: Utilisation de réseau de neurone convolutif pour MNIST
image: assets/images/pic11.jpg
nav-menu: true
---


# Mnist

<img src="assets/images/mnist.jpeg" width="50%" height="50%"/>

Dans cette exemple nous utilisons le jeu de donnée Mnist (disponible à l'adresse suivante https://github.com/zalandoresearch/fashion-mnist) pour apprendre a classer  des images de chiffres de 0 à 9 écrits à la main.

Cette base de données est composé 60 000 images pour faire l'apprentissage et 10000 images pour tester notre algorithme d'apprentissage, Les images sont réparties en 10 classes qui correspondent à 10 chiffres différents: dans l'image suivante nous montrons un exemple des 10 types de classes.



## Developpement

On utilise la librairie pytorch qui permet de créer des réseaux de neurones de manière optimisé et qui à déjà intégré l'ensemble des formules mathématiques qui permettent de créer un réseau de neurone et de faire son apprentissage.
Nous utilisons la librairie torchvision également afin qu'on puisse envoyer les données dans le réseau de neurone


```python
import numpy as np
from torchvision.datasets import MNIST
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

train_dataset = MNIST('./www/data/mnist', train=False, download=True,
                             transform=transformation)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=taille_batch, shuffle=True)

test_dataset = MNIST('./www/data/mnist', train=False, download=True,
                            transform=transformation)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=taille_batch, shuffle=True)
```


```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
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
            ax.imshow(img.reshape(28,28), cmap='gray' if output_class == target else 'autumn')
            ax.axis('off')
            ax.set_title('chiffre {}:  {:.1%}'.format(output_class, prob))
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

    /home/nacim/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:23: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number


    Epoch 0:	 erreur 0.8819	precision 81.25%



<img src="assets/images/mnist/conv/output_6_2.png"  width="50%" height="50%"/>


    Test le réseau sur des données test qui n'on jamais été observé par le réseau
    Accuracy: 96.88%



<img src="assets/images/mnist/conv/output_6_4.png"  width="50%" height="50%"/>


    Accuracy: 95.31%



<img src="assets/images/mnist/conv/output_6_6.png"  width="50%" height="50%"/>


    Accuracy: 98.44%



<img src="assets/images/mnist/conv/output_6_8.png"  width="50%" height="50%"/>


    Accuracy: 95.31%



<img src="assets/images/mnist/conv/output_6_10.png"  width="50%" height="50%"/>


    Accuracy: 96.88%



<img src="assets/images/mnist/conv/output_6_12.png"  width="50%" height="50%"/>


    Accuracy: 95.31%



<img src="assets/images/output_6_14.png"  width="50%" height="50%"/>



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

    /home/nacim/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:23: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number


    Epoch 0:	 erreur 0.8454	precision 87.50%
    Epoch 5:	 erreur 0.2897	precision 81.25%
    Epoch 10:	 erreur 0.2290	precision 75.00%
    Epoch 15:	 erreur 0.2154	precision 87.50%
    Epoch 20:	 erreur 0.1943	precision 100.00%
    Epoch 25:	 erreur 0.1766	precision 81.25%
    Epoch 30:	 erreur 0.1818	precision 87.50%
    Epoch 35:	 erreur 0.1668	precision 93.75%
    Epoch 40:	 erreur 0.1530	precision 100.00%
    Epoch 45:	 erreur 0.1494	precision 93.75%
    Epoch 50:	 erreur 0.1497	precision 100.00%
    Epoch 55:	 erreur 0.1443	precision 100.00%
    Epoch 60:	 erreur 0.1305	precision 100.00%
    Epoch 65:	 erreur 0.1386	precision 93.75%
    Epoch 70:	 erreur 0.1303	precision 87.50%
    Epoch 75:	 erreur 0.1284	precision 93.75%
    Epoch 80:	 erreur 0.1341	precision 100.00%
    Epoch 85:	 erreur 0.1347	precision 93.75%
    Epoch 90:	 erreur 0.1247	precision 100.00%
    Epoch 95:	 erreur 0.1256	precision 93.75%
    Epoch 100:	 erreur 0.1151	precision 93.75%
    Epoch 105:	 erreur 0.1134	precision 100.00%
    Epoch 110:	 erreur 0.1156	precision 100.00%
    Epoch 115:	 erreur 0.1113	precision 100.00%



<img src="assets/images/output_7_2.png"  width="50%" height="50%"/>


    Test le réseau sur des données test qui n'on jamais été observé par le réseau
    Accuracy: 100.00%



<img src="assets/images/mnist/conv/output_7_4.png"  width="50%" height="50%"/>


    Accuracy: 100.00%



<img src="assets/images/mnist/conv/output_7_6.png"  width="50%" height="50%"/>


    Accuracy: 100.00%



<img src="assets/images/mnist/conv/output_7_8.png"  width="50%" height="50%"/>


    Accuracy: 100.00%



<img src="assets/images/mnist/conv/output_7_10.png"  width="50%" height="50%"/>


    Accuracy: 96.88%



<img src="assets/images/mnist/conv/output_7_12.png"  width="50%" height="50%"/>


    Accuracy: 100.00%



<img src="assets/images/mnist/conv/output_7_14.png"  width="50%" height="50%"/>
