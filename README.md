# ml2_project

## Project Description

Every student is asked to develop an original machine learning project, using the methods and approaches presented in the course. At least the following minimal elements need to be part of it:

* Collection or synthetic creation of a data set
* Training/Fine-tuning of one (or more) model(s) (self-developed or pre-trained) or zero-shot/few shots inference from transformer models (visual) or prompt engineering for transformers models (language).
* Interpretation and validation of results + model performance measure

Data domain can be textual, visual or a combination of both (multimodal). The topic and goal of the project can be freely chosen (hint: start from a problem, not from the solution!) and can leverage any NLP or Computer Vision supervised learning- or generative- methods or a combination of these.

## Problem

### Introduction

This is me and my girlfriend Lea (This isn't the problem - Isn't she gorgeous? 😍)

<img src="resources/cuteness_overflow.jpeg"  width="40%" height="40%">

I'd love to delve into a heartwarming story about us, but unfortunately this isn't the goal from this project. But don't hesitate contacting me by mail if you want to know more about us [eglisi1](mailto:<eglisi1@students.zhaw.ch>).

### The real problem

So, here's the scoop. Lea and I have an exciting summer vacation planned in Italy. Ah, Italy! The land of pasta, gelato, and impossibly narrow alleyways. But here's the hitch - we're both about as decisive as a squirrel in the middle of the road. With so many choices, deciding what to do in Italy is proving as challenging as eating spaghetti with a spoon.

That's where you come in, dear machine learning project. We'll be wrangling with the `515K Hotel Reviews Data in Europe` dataset to spare us the agony of choosing the perfect hotel. Because, let's be honest, after figuring out whether to visit the Colosseum or the Leaning Tower of Pisa first, who has the energy left to choose a hotel?

## Solving the problem

### Creating the model

To tackle my problem i wasn't to make a sentiment analysis about the reviews of the hotels. The goal is to find out if the reviews are positive or negative. This will help us to find the perfect hotel for our vacation.
I wanted to make my own model. After I was able to fit a basic model in `01_create_sentiment_model/01_create_sentiment_model.ipynb` I had to fine tune my model.
This isn't my strentgh and I'm kind of lazy. Fortunately I'm able to automate easy python tasks.
So I wrote a python script which will create a model for each config file in the `configs/model` directory.
The script is in the `01_create_sentiment_model` directory and is called `01_create_sentiment_model.py`.
The created model will be saved in the `models` directory.
With this approach I can make some experiments with different models and configs. And pick the best one after the training.
So the creation of the model was quite easy, because I can automatically create a ton of models.
Unfortunately I still have to choose the best one by myself

### Interpretation and validation of results + model performance measure
