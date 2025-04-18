# src/tuning/smac_runner.py

import os
import warnings
import tensorflow as tf
import numpy as np
from smac import Scenario, HyperparameterOptimizationFacade, MultiFidelityFacade
from smac.initial_design import RandomInitialDesign
from smac.intensifier import Hyperband
from smac.optimizer.objective import average_cost
from smac.runhistory.encoder import RunHistoryEncoder
from smac.multi_objective.parego import ParEGO
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import Categorical, UniformFloatHyperparameter, UniformIntegerHyperparameter

from models.base_lstm import build_model
from data.load_data import load_preprocessed_data
from models.evaluate_hparams import run_test

# Suppress TensorFlow & absl logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def smac_objective(cfg):
    """Objective function for SMAC — trains and evaluates a model using the current config."""
    hparams = {
        "layers": [200],
        "time_steps": 5,
        "learning_rate": cfg["learning_rate"],
        "optimizer": cfg["optimizer"],
        "dropout_rate": 0.2,
        "batch_size": cfg["batch_size"],
        "epochs": 20,
        "replicates": 3  # to speed up SMAC trials
    }

    result = run_test(hparams, use_scaler=True, verbose=0)
    return result["mae"]  # SMAC minimizes by default


def run_smac():
    cs = ConfigurationSpace(seed=42)
    cs.add_hyperparameters([
        Categorical("optimizer", ["adam", "adagrad", "nadam"]),
        UniformFloatHyperparameter("learning_rate", lower=0.001, upper=0.1, log=True),
        Categorical("batch_size", [4, 8, 16])
    ])

    scenario = Scenario({
        "run_objective": "quality",
        "cs": cs,
        "output_directory": "outputs/smac_tuner",
        "deterministic": True,
        "n_trials": 10,  # you can increase this
    })

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=smac_objective,
    )

    incumbent = smac.optimize()

    print("\n🎯 Best Hyperparameters Found:")
    for key, val in incumbent.get_dictionary().items():
        print(f"{key}: {val}")

    best_params = {
        "layers": [200],
        "time_steps": 5,
        "learning_rate": incumbent["learning_rate"],
        "optimizer": incumbent["optimizer"],
        "dropout_rate": 0.2,
        "batch_size": incumbent["batch_size"],
        "epochs": 20,
        "replicates": 10
    }

    run_test(best_params)


if __name__ == "__main__":
    run_smac()
