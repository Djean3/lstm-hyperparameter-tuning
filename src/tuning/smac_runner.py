import os
import warnings
import numpy as np
from smac import Scenario, BlackBoxFacade
from smac.initial_design.random_design import RandomInitialDesign
from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer
from data.load_data import load_preprocessed_data
from models.base_lstm import build_model
from models.evaluate_hparams import run_test
#from smac.runhistory.runhistory import RunInfo, RunValue
#from smac.runhistory.status_type import StatusType
from tqdm import tqdm
import time
#start_time = time()

# Silence logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def build_lstm_from_config(cfg, seed: int = 0):
    """Objective function for SMAC."""
    # Optional: set seeds for reproducibility
    import random
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train, X_val, y_val, *_ = load_preprocessed_data()

    model, _ = build_model(
        layers=[200],
        time_steps=X_train.shape[1],
        num_features=X_train.shape[2],
        optimizer_name=cfg["optimizer"],
        learning_rate=cfg["learning_rate"],
        dropout_rate=0.2,
        use_early_stopping=False
    )
    model.compile(optimizer=model.optimizer, loss="mean_squared_error", metrics=["mae"])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=cfg["batch_size"],
        epochs=10,
        verbose=0
    )
    return float(np.min(history.history["val_mae"]))

def run_smac_tuner():
    cs = ConfigurationSpace(seed=42)
    cs.add_hyperparameters([
        Categorical("optimizer", ["adam", "adagrad", "nadam"]),
        Float("learning_rate", (0.001, 0.1), log=True),
        Integer("batch_size", (4, 16)),
    ])

    scenario = Scenario(cs, walltime_limit=600, output_directory="outputs/smac_tuner")

    initial_design = RandomInitialDesign(scenario, n_configs=5)

    smac = BlackBoxFacade(
        scenario=scenario,
        target_function=build_lstm_from_config,
        initial_design=initial_design
    )
 #######################################
    # BUILD PROGRESS BAR
##########################################
    incumbent = smac.optimize()
    print("\n🎯 Best Hyperparameters Found:")
    for k, v in incumbent.items():
        print(f"{k}: {v}")

    best_params = {
        "layers": [200],
        "time_steps": load_preprocessed_data()[0].shape[1],
        "learning_rate": incumbent["learning_rate"],
        "optimizer": incumbent["optimizer"],
        "dropout_rate": 0.2,
        "batch_size": incumbent["batch_size"],
        "epochs": 20,
        "replicates": 10
    }

    run_test(best_params)

if __name__ == "__main__":
    run_smac_tuner()
