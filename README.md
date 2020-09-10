# Smart-Schematic
this project intends to color the text of transistors in an analog circuit.  
to perform such a method I've used open-cv with OCR.  
first I've used open-cv deep learning paper (Natural scene text understanding) by Céline Mancas-Thillou, Bernard Gosselin.  
which showed great success in text localization.  
So I'm using it, and I've followed this tutorial with some edits, to fit my project.  
to make it happens (https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/) (I'm c++ fan ^_^).  
after localizing the text, I've performed OCR to detect the words.  
If i've used OCR on its own, it should great failure in noisy image, thats why I've preferred to use EAST with it.  

## Table of contents:
* [Application](#system-layout).  
* [Features](#features).  
* [Limitations](#limitations).  
* [Future thoughts](#future-thoughts).  

## Application:
System layout  
![schem](https://user-images.githubusercontent.com/31229408/92678203-c1dc4800-f325-11ea-9414-940e321b0d8a.PNG)  

Output  
![out](https://user-images.githubusercontent.com/31229408/92678242-d7517200-f325-11ea-9e67-149eae6a0619.PNG)  

##Features:

    1- you can select whatever the transistor you like from a drop down list.  
    2- transistor selected will be colored.  
    3- simple GUI.  

##Limitations:

    1- if there were many words, or complex structure, it will fail detecting the right place, and words.  

## Future thoughts:

    1- improve architecture.  
    2- make it global.  
    
