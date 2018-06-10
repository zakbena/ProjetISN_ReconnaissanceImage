---
layout: page
title: QuickDraw
nav-menu: true
---

<div id="main" class="alt">


<section id="one">
<div class="inner">
<h1> Quickdraw</h1>

Google a developpé Quickdraw (https://quickdraw.withgoogle.com/), un jeu similaire à Pictionnary: à partir d'un mot, il faut le dessiner et l'application essaie de prédire correctement ce que représente l'image.
Grâce à ce jeu, Google à collecter plus de 3 millions d'images afin de faire de la recherche en intelligence artificiel et machine learning.

Dans ce mini projet, nous avons utiliser les d'une dixaine de types d'image et nous avons utiliser un réseau de neurone afin d'apprendre  à reconnaitre des dessins  de ces 10 types de classes.

De la même maniere que pour Fashion Mnist et CIFAR10, nous avons utiliser des réseaux de neurones convolutif et nous avons testé differents réseaux de neurones. Pour réussir, à apprendre les images, nous avons dûe testé de nouvelles architectures de réseaux de neurones, et l'approche qui a fonctionné est Resnet34.


<h1> Resnet 34 </h1>
Resnet est une architecture de réseaux de neurones de sorte qu'il y a 34 couches de neurone dont 33 couche de réseaux de neurones convolutifs. Contrairement à VGG19, Les sorties des couches sont propager 2 couches apprès de sorte que la sortie des couches permet de conserver l'information  des couches de neurones du début.
<div class="box alt">
	<div class="row 50% uniform">
<div class="12u"><span class="image fit"><img src="assets/images/exemple_resnet34.jpg" alt="" /></span></div>


</div>
</div>


<h1> Résultat après apprentissage </h1>
Nous avons utiliser Resnet34  pour faire l'apprentissage en utilisant 100 epoch pour l'apprentissage. Nous avons ensuite tester sur des images qui n'ont jamais été utilisé  pendant l'apprentissage.

<div class="box alt">
	<div class="row 50% uniform">
<div class="4u"><span class="image fit"><img src="assets/images/quickdraw/batch_0.png" width="50%" height="50%" alt="" /></span></div>
<div class="4u"><span class="image fit"><img src="assets/images/quickdraw/batch_1.png" width="50%" height="50%" alt="" /></span></div>
<div class="4u"><span class="image fit"><img src="assets/images/quickdraw/batch_2.png" width="50%" height="50%" alt="" /></span></div>
		<!-- Break -->
<div class="4u"><span class="image fit"><img src="assets/images/quickdraw/batch_3.png" width="50%" height="50%" alt="" /></span></div>
<div class="4u"><span class="image fit"><img src="assets/images/quickdraw/batch_4.png" width="50%" height="50%" alt="" /></span></div>
<div class="4u"><span class="image fit"><img src="assets/images/quickdraw/batch_5.png" width="50%" height="50%" alt="" /></span></div>
		<!-- Break -->
<div class="4u"><span class="image fit"><img src="assets/images/quickdraw/batch_6.png" width="50%" height="50%" alt="" /></span></div>
<div class="4u"><span class="image fit"><img src="assets/images/quickdraw/batch_7.png" width="50%" height="50%" alt="" /></span></div>
<div class="4u"><span class="image fit"><img src="assets/images/quickdraw/batch_8.png" width="50%" height="50%" alt="" /></span></div>
	</div>
</div>

<h1> Demo Live </h1>
Afin de tester notre réseau de neurone en condition réel, nous avons réalisé une petite application web développé en python avec pytorch pour créer le réseau de neurone et une page web qui permet de communiquer avec le page web grâce à des requêtes HTTP. Le code python tourne en continue et attend d'avoir une requete HTTP pour faire une prédiction. La prédiction est calculé à a chaque fois que le canvas  est modifié: la page web appelle le code python via une requete http en envoyant les un tableau qui représente l'ensemble des pixels de l'image que l'on déssine dans le canvas HTML5.
<div class="box alt">
<div class="row 50% uniform">
<iframe src="https://quickdraw-10-classification.herokuapp.com/" name="frame2" frameborder="0" scrolling="auto" onload="" allowtransparency="True" width="100%" height="800"></iframe>
</div>
</div>





</div>
</section>
</div>

