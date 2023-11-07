import optuna
import numpy as np
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
import joblib
import argparse
import sys
from objective_func import objective, set_seed, print_best_callback


def main():
  
    parser = argparse.ArgumentParser(description=("arguments")) 
    parser.add_argument("--epochs", default="10")
    parser.add_argument("--patience_epochs", default="10")
    parser.add_argument("--trails", default="10")
    parser.add_argument("--seed", default="12345")
    parser.add_argument("--load_study", default="False")
    parser.add_argument("--dataset_file", default="dataset.npy")
    parser.add_argument("--study_file", default="study.pkl")

    
    args = parser.parse_args()
    dataset_file = args.dataset_file
    study_file = args.study_file
    epochs = np.int32(args.epochs)
    patience_epochs = np.int32(args.patience_epochs)
    trails = np.int32(args.trails)
    seed = np.int32(args.seed)
    epochs = np.int32(args.epochs)
    
    print('epochs, trails,  patience_epochs = ', epochs, trails, patience_epochs)

    if args.load_study=="True":
        print('loading pickle file', study_file)
        study = joblib.load(study_file)
        
        print("Best trial until now:")
        print(" Value: ", study.best_trial.value)
        print(" Params: ")
        for key, value in study.best_trial.params.items():
            print(f"{key}: {value}")

    else:
        print('created study file')
        study = optuna.create_study(direction="minimize")
    
    set_seed(seed)

    with open(dataset_file, 'rb') as f:
        print(dataset_file)
        xdata = np.load(f)
        ydata = np.load(f)
        weights = np.load(f)

    def print_best_wrapper(study, trial):
        return print_best_callback(study, trial, study_file)

    def wrapper(trial):
        return objective(trial, xdata, ydata, weights, epochs, patience_epochs, study_file, seed)
    
    study.optimize(wrapper, n_trials=trails, callbacks=[print_best_wrapper])

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))
   
###############################################################################

if __name__ == "__main__":
    # Check the number of arguments passed
    if len(sys.argv) > 20:
        print("Too many arguments..")
    else:
        main()