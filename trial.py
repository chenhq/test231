from hyperopt import fmin, tpe, Trials, space_eval
try:
    import _pickle as pickle
except:
    import pickle


def run_a_trial(trials_file, objective, space):
    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 1

    print("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(trials_file, "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open(trials_file, "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")
    best_params = space_eval(space, best)
    print("best_params: {}".format(best_params))


