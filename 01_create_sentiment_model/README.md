# Structure

In this directory, we will create a sentiment model using the IMDB dataset.
For a detailed explanation have a look at the notebook `01_create_sentiment_model.ipynb`.

In the python file `01_create_sentiment_model.py` we will create a model for each config file in the `configs/model` directory.
The model will be saved in the `models` directory.
In order to run the script, you need to call it from the root directory of the project.

```bash
python 01_create_sentiment_model/01_create_sentiment_model.py
```

To ignore a config file, add a `zzz` in front of the file name.
