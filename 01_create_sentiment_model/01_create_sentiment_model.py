# ML stuff
from keras.datasets import imdb
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split

# General stuff
from typing import Tuple
import logging
import numpy as np
import yaml
import os

# load config
config_dir = os.path.join(os.getcwd(), "config")
with open(os.path.join(config_dir, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Define global variables
num_words = config["num_words"]
model_config = {}

# Set up logging
logging.basicConfig(
    level=config["logging"]["level"],
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_model() -> None:
    test_size = model_config["model"]["training"]["test_size"]
    random_state = model_config["model"]["training"]["random_state"]

    data, targets = load_data()
    data = vectorize(data)
    targets = np.array(targets).astype("float32")
    train_x, test_x, train_Y, test_Y = train_test_split(data, targets, test_size=test_size, random_state=random_state)
    model = define_model()
    model = compile_model(model)
    model = fit_model(model, train_x, train_Y, test_x, test_Y)
    evaluate_model(model, test_x, test_Y)
    persist_model(model)


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=num_words)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    logger.info("Loaded data")
    return data, targets


def vectorize(sequences: np.ndarray, dimension=num_words) -> np.ndarray:
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    logger.info("Vectorized data")
    return results


def define_model() -> models.Sequential:
    model = models.Sequential()
    activation = model_config["model"]["training"]["activation_function"]
    model.add(layers.Dense(units=50, activation=activation, input_shape=(num_words,)))
    model.add(layers.Dropout(rate=0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(units=50, activation=activation))
    model.add(layers.Dropout(rate=0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(units=50, activation=activation))
    model.add(layers.Dense(units=1, activation="sigmoid"))
    logger.info("Defined model")
    return model


def compile_model(model: models.Sequential) -> models.Sequential:
    optimizer = model_config["model"]["compile"]["optimizer"]
    loss = model_config["model"]["compile"]["loss_function"]
    metrics = model_config["model"]["compile"]["metrics"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def fit_model(model: models.Sequential, train_x: np.ndarray, train_Y: np.ndarray, test_x: np.ndarray, test_Y: np.ndarray) -> models.Sequential:
    epochs = model_config["model"]["training"]["epochs"]
    batch_size = model_config["model"]["training"]["batch_size"]

    logger.info("Fitting model...")
    verbose = 1 if logging.DEBUG >= logging.root.level else 0
    model.fit(
        train_x,
        train_Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_x, test_Y),
        verbose=verbose,
    )
    return model


def evaluate_model(model: models.Sequential, test_x: np.ndarray, test_Y: np.ndarray) -> models.Sequential:
    scores = model.evaluate(x=test_x, y=test_Y, verbose=0)
    logger.info(f"Accuracy: {scores[1]*100:.2f}%")


def persist_model(model: models.Sequential) -> None:
    filepath = os.path.join(os.getcwd(), config["model_path"], model_config["model"]["name"])
    logger.debug(f"Saving model to {filepath}")
    model.save(filepath=filepath, overwrite=True)


if __name__ == "__main__":
    model_config_dir = os.path.join(config_dir, "model")
    for model_config_file in os.listdir(model_config_dir):
        model_config_file = os.path.join(model_config_dir, model_config_file)
        logger.debug(f"Loading model config file: {model_config_file}")
        with open(model_config_file, "r") as f:
            model_config = yaml.safe_load(f)
            logger.info("========================================")
            logger.info(f"Creating model {model_config['model']['name'].upper()}")
            logger.info("========================================")
            create_model()
