# ml2_project

## Project Description

Every student is asked to develop an original machine learning project, using the methods and approaches presented in the course. At least the following minimal elements need to be part of it:

* Collection or synthetic creation of a data set
* Training/Fine-tuning of one (or more) model(s) (self-developed or pre-trained) or zero-shot/few shots inference from transformer models (visual) or prompt engineering for transformers models (language).
* Interpretation and validation of results + model performance measure

Data domain can be textual, visual or a combination of both (multimodal). The topic and goal of the project can be freely chosen (hint: start from a problem, not from the solution!) and can leverage any NLP or Computer Vision supervised learning- or generative- methods or a combination of these.

## Technical setup

To run the project, you need to install the requirements in the `requirements.txt` file.
If needed craete a virtual environment first.

```bash
python -m venv venv
```

Activate the virtual environment.

```bash
source venv/bin/activate
```

You can do this by running the following command in the root directory of the project:

```bash
pip install -r requirements.txt
```

## Problem

### Introduction

This is me and my girlfriend Lea (That's not the problem - But isn't she gorgeous? 😍)

<img src="resources/cuteness_overflow.jpeg"  width="40%" height="40%">

I'd love to delve into a heartwarming story about us, but unfortunately this isn't the goal from this project. But don't hesitate contacting me by mail if you want to know more about us [eglisi1](mailto:<eglisi1@students.zhaw.ch>).

### The real problem

So, here's the scoop. Lea and I have an exciting summer vacation planned in Italy. Ah, Italy! The land of pasta, gelato, and impossibly narrow alleyways. But here's the hitch - we're both about as decisive as a squirrel in the middle of the road. With so many choices, deciding what to do in Italy is proving as challenging as eating spaghetti with a spoon.

That's where you come in, dear machine learning project. We'll be wrangling with the [515K Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) dataset to spare us the agony of choosing the perfect hotel. Because, let's be honest, after figuring out whether to visit the Colosseum or the Leaning Tower of Pisa first, who has the energy left to choose a hotel?

## Solving the problem

### 01 - Creating the model

To tackle my problem, I wanted to conduct sentiment analysis on hotel reviews. The goal is to discern whether the reviews are positive or negative, which will aid us in finding the perfect hotel for our vacation.
I decided to create my own model. Once I managed to fit a basic model in `01_create_sentiment_mode/01_crate_sentiment_model.ipynb`, I had to fine-tune it. This isn't my strength, and I admit to being somewhat lazy. Fortunately, I can automate routine Python tasks.
Consequently, I wrote a Python script that creates a model for each config file in the`configs/model directory`. The script is located in the `01_create_sentiment_model` directory and is called `01_create_sentiment_model.py`. The generated models are stored in the models directory.
With this approach, I can experiment with different models and configurations and choose the best one after training. The creation of the models was relatively easy since I can automatically generate a multitude of them. Unfortunately, I still have to manually select the best one.

### 02 - Interpretation and validation of results + model performance measure

After I ran the notebook `02_interpretation_and_validation_of_results.ipynb` I realized that vader outperforms my models.
I didn't safe all the configs because the python script would take a while to complete. But I saved the best models and the config file in the `models` respectively in the `config` directory.

The result of the models are compared to the vader model in the notebook `02_interpretation_and_validation_of_results.ipynb`.
Because the dataset is well balanced between positive and negative reviews, I decided to use the accuracy as a performance measure.
In a classification problem I want a better chance than throwing a coin because otherwise I could just flip a coin to decide which hotel to book.

Unfortunately with my basic model I only achieved an accuracy of 0.5. This is not better than flipping a coin.
Vader could achieve an accuracy of 0.7. This is better than flipping a coin but still not good enough.

All models have in common that it's easier to predict positive reviews than negative reviews.
Even the vader model only managed to predict around 30% of the negative reviews correctly.

### Interpretation of results

The interpretation of the results have some issues with my approach.
The models I created deliever a binary result. Either the review is positive or negative.
But the vader model delievers a score between -1 and 1.
Since I treated the vader model as a binary classifier, I had to define a threshold. Which I defined at 0.
This means that every review with a score above 0 is positive and every review with a score below 0 is negative.

The model `basic_model` does not have the best accuracy but it has a better mix than just predicting everything as positive like the `sigmoid_many_epochs` model.

### Reflection

The project was a lot of fun. I learned a lot about sentiment analysis and how to create a model.
I think that my approach with a basic model and use different hyperparameters to fit is a good approach to gel a lot of models but it's not the best approach to get the best model.
I think that I should have used a pre-trained model and fine-tune it. This would have given me a better result.

### Validation

I validated my results by comparing them to the vader model. I also used the accuracy as a performance measure.
It showed that my models are not better than flipping a coin. But the vader model is better than flipping a coin.
So my models are not really useful for Lea and me to find the perfect hotel.