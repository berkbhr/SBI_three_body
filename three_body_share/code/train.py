import os
# the keras backend must be chosen before bayesflow is imported
os.environ["KERAS_BACKEND"] = "torch"

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
import bayesflow as bf
from device import get_device

# training settings
BATCH_SIZE = 64
EPOCHS = 150
INITIAL_LEARNING_RATE = 5e-4
SUMMARY_DIM = 64
EARLY_STOPPING_PATIENCE = 20

MODEL_PATH = "models/trained_model.keras"
TRAINING_LOG_PATH = "models/training_log.txt"

class EpochLogger(keras.callbacks.Callback):
    # writes one line per epoch to models/training_log.txt as training runs;
    # a keras Callback subclass is needed here because keras is the library
    # that calls on_epoch_end during fit()
    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_train_begin(self, logs=None):
        f = open(self.path, "w")
        f.write("epoch train_loss val_loss\n")
        f.close()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        f = open(self.path, "a")
        f.write(str(epoch + 1) + " " + str(logs.get("loss")) + " " + str(logs.get("val_loss")) + "\n")
        f.close()

def load_split(path):
    # anti-leak rule from data_contracts.md: the network inputs are the noisy
    # observation and the known window offset t0; the inference target is theta.
    # Everything else in the npz (clean trajectory, quality variables) is for
    # diagnostics and is deliberately not loaded here. t0 is reshaped to (N, 1)
    # so the adapter can treat it as a single inference condition.
    data = np.load(path)
    theta = data["theta"]
    obs_noisy = data["obs_noisy"]
    t0 = data["t0"].reshape(-1, 1)
    return {"theta": theta, "obs_noisy": obs_noisy, "t0": t0}

def build_workflow():
    # the adapter standardizes the parameters, the observations and the known
    # window offset t0. It casts to float32, groups theta as the inference
    # target, obs_noisy as the summary input, and t0 as a direct inference
    # condition; the workflow's standardize list normalizes all three groups
    # (obs_noisy is standardized before it reaches the summary network).
    adapter = (
        bf.Adapter()
        .convert_dtype(from_dtype="float64", to_dtype="float32")
        .concatenate(["theta"], into="inference_variables")
        .concatenate(["obs_noisy"], into="summary_variables")
        .concatenate(["t0"], into="inference_conditions")
    )

    # time-aware summary network over the 10 recorded states
    summary_network = bf.networks.TimeSeriesNetwork(summary_dim=SUMMARY_DIM)

    # flow matching inference network, chosen for multimodal posteriors
    inference_network = bf.networks.FlowMatching()

    workflow = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=summary_network,
        inference_network=inference_network,
        standardize=["inference_variables", "summary_variables", "inference_conditions"],
        initial_learning_rate=INITIAL_LEARNING_RATE,
    )
    return workflow

def has_nan(values):
    for v in values:
        if np.isnan(v):
            return True
    return False

def check_gate(train_loss, val_loss):
    # gate from the owner: stop and report if the loss goes NaN or the
    # validation loss never decreases. Returns a list of problem strings,
    # empty if the run is healthy.
    problems = []
    if has_nan(train_loss) or has_nan(val_loss):
        problems.append("loss went NaN during training")
    if min(val_loss) >= val_loss[0]:
        problems.append("validation loss never decreased below its first epoch value")
    return problems

def save_history_plot(train_loss, val_loss, path):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training and validation loss")
    plt.legend()
    plt.savefig(path)
    plt.close()

def write_training_info(path, device, walltime, n_train, train_loss, val_loss, best_epoch):
    f = open(path, "w")
    f.write("device " + device + "\n")
    f.write("epochs " + str(len(train_loss)) + "\n")
    f.write("batch_size " + str(BATCH_SIZE) + "\n")
    f.write("n_train " + str(n_train) + "\n")
    f.write("initial_learning_rate " + str(INITIAL_LEARNING_RATE) + "\n")
    f.write("lr_schedule cosine_decay_with_warmup\n")
    f.write("optimizer adamw\n")
    f.write("summary_network time_series_network\n")
    f.write("summary_dim " + str(SUMMARY_DIM) + "\n")
    f.write("inference_network flow_matching\n")
    f.write("inference_conditions t0\n")
    f.write("early_stopping_patience " + str(EARLY_STOPPING_PATIENCE) + "\n")
    f.write("walltime_seconds " + str(walltime) + "\n")
    f.write("final_train_loss " + str(train_loss[-1]) + "\n")
    f.write("final_val_loss " + str(val_loss[-1]) + "\n")
    f.write("best_val_loss " + str(min(val_loss)) + "\n")
    f.write("saved_model_is_best_validation_epoch True\n")
    f.write("best_epoch " + str(best_epoch) + "\n")
    f.close()

def main():
    device = get_device()

    train = load_split("data/train.npz")
    validation = load_split("data/validation.npz")
    n_train = train["theta"].shape[0]
    print("loaded", n_train, "training and", validation["theta"].shape[0], "validation simulations")

    workflow = build_workflow()

    # TerminateOnNaN stops the run immediately if the loss becomes NaN, so a
    # broken run does not waste the full training time before the gate sees it
    nan_callback = keras.callbacks.TerminateOnNaN()

    # saves the whole model to MODEL_PATH only when validation loss improves,
    # so by the end of training the file on disk already holds the
    # best-validation weights, not the final epoch's
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH, monitor="val_loss", mode="min", save_best_only=True
    )

    # stops after EARLY_STOPPING_PATIENCE epochs with no validation
    # improvement, and restores the in-memory model to its best weights
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
    )

    epoch_logger = EpochLogger(TRAINING_LOG_PATH)

    start_time = time.time()
    history = workflow.fit_offline(
        data=train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=validation,
        callbacks=[nan_callback, checkpoint_callback, early_stopping_callback, epoch_logger],
    )
    walltime = time.time() - start_time
    print("training finished in", walltime, "seconds on", device)

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    problems = check_gate(train_loss, val_loss)
    if not os.path.exists(MODEL_PATH):
        problems.append("model file did not save")
    if len(problems) > 0:
        print("GATE FAILED:")
        for p in problems:
            print(" ", p)
        print("final train loss:", train_loss[-1])
        print("final val loss:", val_loss[-1])
        return

    # the checkpoint callback already wrote the best-validation model to
    # MODEL_PATH during training; nothing more to save here
    print("models/trained_model.keras holds the best-validation weights")

    best_epoch = val_loss.index(min(val_loss)) + 1

    save_history_plot(train_loss, val_loss, "models/training_history.png")
    print("saved models/training_history.png")

    write_training_info(
        "models/training_info.txt", device, walltime, n_train, train_loss, val_loss, best_epoch
    )
    print("saved models/training_info.txt")

    print("device:", device)
    print("wall time seconds:", walltime)
    print("epochs run:", len(train_loss))
    print("final train loss:", train_loss[-1])
    print("final val loss:", val_loss[-1])
    print("best val loss:", min(val_loss))
    print("best epoch:", best_epoch)

if __name__ == "__main__":
    main()
