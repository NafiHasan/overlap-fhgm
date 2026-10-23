"""
Classifiers: the learned pattern encoder with its network head, and the plain
network and linear SVM used without the encoder.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
import logging
import random
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from .validation import validate_head

logger = logging.getLogger(__name__)


def _set_training_seed(config):
    """Seed model construction and training, and optionally require deterministic ops."""
    seed = int(config.get("random_state", 42))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if config.get("deterministic_training", False):
        # TensorFlow raises if a requested operation has no deterministic
        # implementation.  In strict mode that is preferable to silently
        # producing a run that cannot be reproduced on the same platform.
        tf.config.experimental.enable_op_determinism()

    return seed


class NN:
    """
    Neural network classifier using TensorFlow/Keras.

    Builds a Sequential model with configurable hidden layers, each followed
    by layer normalization and dropout, with optional L2 regularization.

    Args:
        config (dict): Classification configuration dictionary. Expected keys:
            nn_layers, nn_dropouts, nn_l2_regs, nn_learning_rate,
            nn_epochs, nn_batch_size, nn_patience, num_classes.
    """

    def __init__(self, config):
        validate_head(config)
        tf.keras.backend.clear_session()
        self.config = config
        self.seed = _set_training_seed(config)
        layer_sizes = config["nn_layers"]
        dropouts = config["nn_dropouts"]
        l2_regs = config["nn_l2_regs"]
        num_classes = config["num_classes"]

        model_layers = []
        for i, (size, dropout, l2) in enumerate(zip(layer_sizes, dropouts, l2_regs)):
            kwargs = {"activation": "relu"}
            if l2 > 0:
                kwargs["kernel_regularizer"] = tf.keras.regularizers.l2(l2)
            model_layers.append(layers.Dense(size, **kwargs))
            model_layers.append(layers.LayerNormalization())
            model_layers.append(layers.Dropout(dropout))

        model_layers.append(layers.Dense(num_classes, activation="softmax"))
        self.model = tf.keras.Sequential(model_layers)

    def fit(self, X_train, y_train, X_val, y_val):
        """Fit the network; the validation data drive early stopping and the learning rate."""
        seed = _set_training_seed(self.config)
        logger.info(
            "  Classifier seed=%d, deterministic_ops=%s",
            seed,
            bool(self.config.get("deterministic_training", False)),
        )

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config["nn_learning_rate"]),
            metrics=["accuracy"],
        )

        X_train_split, y_train_split = X_train, y_train
        num_classes = self.config["num_classes"]
        y_train_cat = to_categorical(y_train_split, num_classes=num_classes)
        y_val_cat = to_categorical(y_val, num_classes=num_classes)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=self.config["nn_patience"], restore_best_weights=True
        )

        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=self.config.get("nn_lr_decay_factor", 0.5),
            patience=self.config.get("nn_lr_patience", 5),
            min_lr=self.config.get("nn_min_lr", 1e-5),
            verbose=0,
        )

        # Print a summary every N epochs instead of every epoch
        log_interval = self.config.get("nn_log_interval", 10)

        class EpochLogger(tf.keras.callbacks.Callback):
            """Print metrics every `interval` epochs."""
            def __init__(self, interval):
                super().__init__()
                self.interval = interval

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.interval == 0 or epoch == 0:
                    acc = logs.get("accuracy", 0)
                    val_acc = logs.get("val_accuracy", 0)
                    loss = logs.get("loss", 0)
                    val_loss = logs.get("val_loss", 0)
                    logger.debug(f"Epoch {epoch+1}: "
                          f"acc={acc:.4f}, val_acc={val_acc:.4f}, "
                          f"loss={loss:.4f}, val_loss={val_loss:.4f}")

        history = self.model.fit(
            X_train_split,
            y_train_cat,
            epochs=self.config["nn_epochs"],
            batch_size=self.config["nn_batch_size"],
            callbacks=[early_stopping, lr_scheduler, EpochLogger(log_interval)],
            validation_data=(X_val, y_val_cat),
            shuffle=self.config.get("nn_shuffle", True),
            verbose=0,
        )

        # Single console summary line for training progress
        total_epochs = len(history.history["accuracy"])
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        best_val_acc = history.history["val_accuracy"][best_epoch - 1]
        logger.info(f"  Training done: {total_epochs} epochs, "
                     f"best_val_acc={best_val_acc:.4f} (epoch {best_epoch})")

        self.history = history
        return history

    def predict(self, X_test):
        preds = self.model(X_test, training=False).numpy()
        prediction = np.argmax(preds, axis=1)
        return list(prediction)


class SupervisedPatternSetNN(NN):
    """Jointly learn pattern embeddings and the graph-classification head.

    Each graph is represented by a row of scaled relative frequencies, one per
    pattern. A bias-free dense layer therefore computes a weighted sum of
    learned pattern embeddings. The pooled vector is layer-normalized,
    augmented with log(1 + number of present patterns), and passed through the
    same configurable MLP head used by :class:`NN`.

    Pattern dropout is applied only to the pooling branch during training. The
    count of present patterns always uses the original input, so it is exact.
    """

    def __init__(self, config, input_dim):
        validate_head(config)
        tf.keras.backend.clear_session()
        self.config = config
        self.seed = _set_training_seed(config)

        input_dim = int(input_dim)
        self.input_dim = input_dim
        embedding_dim = int(config["embedding_dim"])
        pattern_dropout = float(config.get("supervised_pattern_dropout", 0.1))
        encoder_l2 = float(config.get("supervised_encoder_l2", 0.001))
        # False drops every LayerNormalization (encoder and head).
        layer_norm = config.get("supervised_layer_norm")
        layer_norm = True if layer_norm is None else layer_norm
        if type(layer_norm) is not bool:
            raise ValueError("supervised_layer_norm must be true or false")
        if input_dim <= 0:
            raise ValueError("Supervised pattern encoder requires at least one pattern")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0 <= pattern_dropout < 1:
            raise ValueError("supervised_pattern_dropout must be in [0, 1)")
        if encoder_l2 < 0:
            raise ValueError("supervised_encoder_l2 must be non-negative")

        pattern_input = tf.keras.Input(
            shape=(input_dim,), name="pattern_presence"
        )
        dropped_patterns = layers.Dropout(
            pattern_dropout,
            seed=self.seed,
            name="pattern_dropout",
        )(pattern_input)
        encoder_kwargs = {
            "units": embedding_dim,
            "use_bias": False,
            "name": "pattern_sum",
        }
        if encoder_l2 > 0:
            encoder_kwargs["kernel_regularizer"] = tf.keras.regularizers.l2(
                encoder_l2
            )
        pooled = layers.Dense(**encoder_kwargs)(dropped_patterns)
        if layer_norm:
            pooled = layers.LayerNormalization(name="pattern_sum_norm")(pooled)
        # "nonzero" adds log(1 + number of present patterns); "none" leaves it out.
        count_feature = config.get("supervised_count_feature") or "nonzero"
        if count_feature == "nonzero":
            count_values = lambda values: tf.reduce_sum(
                tf.cast(tf.not_equal(values, 0), values.dtype), axis=1, keepdims=True)
        elif count_feature != "none":
            raise ValueError("supervised_count_feature must be 'nonzero' or 'none'")
        representation_parts = [pooled]
        if count_feature != "none":
            log_pattern_count = layers.Lambda(
                lambda values: tf.math.log1p(count_values(values)),
                output_shape=(1,),
                name="log_pattern_count",
            )(pattern_input)
            representation_parts.append(log_pattern_count)
        if len(representation_parts) > 1:
            hidden = layers.Concatenate(name="graph_representation")(
                representation_parts
            )
        else:
            hidden = representation_parts[0]

        for size, dropout, l2 in zip(
                config["nn_layers"], config["nn_dropouts"],
                config["nn_l2_regs"]):
            head_kwargs = {"activation": "relu"}
            if l2 > 0:
                head_kwargs["kernel_regularizer"] = tf.keras.regularizers.l2(l2)
            hidden = layers.Dense(size, **head_kwargs)(hidden)
            if layer_norm:
                hidden = layers.LayerNormalization()(hidden)
            hidden = layers.Dropout(dropout)(hidden)

        prediction = layers.Dense(
            config["num_classes"], activation="softmax", name="class_prediction"
        )(hidden)
        self.model = tf.keras.Model(
            inputs=pattern_input,
            outputs=prediction,
            name="supervised_pattern_set_nn",
        )
        logger.info(
            "  Supervised pattern encoder: %d patterns -> %d dims, "
            "pattern_dropout=%.3f, encoder_l2=%.6f",
            input_dim,
            embedding_dim,
            pattern_dropout,
            encoder_l2,
        )


def plot_training_history(history, save_path):
    """
    Plot training and validation accuracy/loss curves and save as PNG.

    Args:
        history: Keras History object from model.fit().
        save_path: Absolute path to save the PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless servers
    import matplotlib.pyplot as plt

    hist = history.history
    epochs = range(1, len(hist["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy plot ---
    ax1.plot(epochs, hist["accuracy"], "C0-o", markersize=3, label="Train Accuracy")
    ax1.plot(epochs, hist["val_accuracy"], "C1-o", markersize=3, label="Val Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training & Validation Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Loss plot ---
    ax2.plot(epochs, hist["loss"], "C0-o", markersize=3, label="Train Loss")
    ax2.plot(epochs, hist["val_loss"], "C1-o", markersize=3, label="Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training & Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add best epoch marker
    best_epoch = np.argmin(hist["val_loss"]) + 1
    best_val_acc = hist["val_accuracy"][best_epoch - 1]
    fig.suptitle(
        f"Best val_loss at epoch {best_epoch} (val_acc={best_val_acc:.4f})",
        fontsize=10, y=0.02,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training plot saved: {save_path}")


class LinearSVM:
    """
    Linear SVM classifier using sklearn's LinearSVC.

    Uses one-vs-rest for multi-class.

    Args:
        config (dict): Classification configuration dictionary. Relevant keys:
            random_state, svm_C, svm_max_iter.
    """

    def __init__(self, config):
        from sklearn.svm import LinearSVC

        self.config = config
        seed = int(config.get("random_state", 42))

        self.model = LinearSVC(
            penalty="l2",
            C=config.get("svm_C", 1.0),
            max_iter=config.get("svm_max_iter", 5000),
            random_state=seed,
            dual="auto",
        )
        # No training history, so no training plot.
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit the model. Validation data is accepted but unused."""
        self.model.fit(X_train, y_train)
        logger.info("  LinearSVM done: C=%.4f", self.config.get("svm_C", 1.0))
        return None

    def predict(self, X_test):
        return list(self.model.predict(X_test))
