"""Handwritten Digit Recognition using CNN - training and evaluation script."""
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
BATCH = 128
EPOCHS = 20
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
keras.utils.set_random_seed(SEED)

# ---------------------------------------------------------------- data
d = np.load("data/mnist.npz")
x_all, y_all, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
x_train, x_val, y_train, y_val = train_test_split(
    x_all, y_all, test_size=0.10, stratify=y_all, random_state=SEED
)
add_ch = lambda a: a[..., None].astype("float32")
x_train, x_val, x_test_c = add_ch(x_train), add_ch(x_val), add_ch(x_test)
print("train/val/test:", len(x_train), len(x_val), len(x_test_c))

counts = {
    "train": np.bincount(y_train, minlength=10).tolist(),
    "val": np.bincount(y_val, minlength=10).tolist(),
    "test": np.bincount(y_test, minlength=10).tolist(),
}

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000, seed=SEED)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test_c, y_test)).batch(BATCH).prefetch(tf.data.AUTOTUNE)

# --------------------------------------------------------------- model
def build_model():
    return keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            # augmentation (active only during training)
            layers.RandomRotation(0.08),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomZoom(0.1),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(10, activation="softmax"),
        ],
        name="digit_cnn",
    )


model = build_model()
model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
total_params = model.count_params()
trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
print("params", total_params, trainable, non_trainable)

cbs = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    keras.callbacks.ModelCheckpoint(f"{OUT}/digit_cnn.keras", monitor="val_loss", save_best_only=True),
]
hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=2)
epochs_run = len(hist.history["loss"])

# ---------------------------------------------------------- evaluation
best = keras.models.load_model(f"{OUT}/digit_cnn.keras")
test_loss, test_acc = best.evaluate(test_ds, verbose=0)
probs = best.predict(x_test_c, batch_size=256, verbose=0)
pred = probs.argmax(1)
acc = accuracy_score(y_test, pred)
p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro")
auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
cm = confusion_matrix(y_test, pred)
rep = classification_report(y_test, pred, digits=4, output_dict=True)
print(classification_report(y_test, pred, digits=4))
print("acc", acc, "macroP", p, "macroR", r, "macroF1", f, "auc", auc, "loss", test_loss)

metrics = {
    "accuracy": acc,
    "precision_macro": p,
    "recall_macro": r,
    "f1_macro": f,
    "roc_auc_ovr_macro": auc,
    "test_loss": float(test_loss),
    "epochs_run": epochs_run,
    "best_epoch": int(np.argmin(hist.history["val_loss"]) + 1),
    "total_params": int(total_params),
    "trainable_params": trainable,
    "non_trainable_params": non_trainable,
    "per_class": {str(k): rep[str(k)] for k in range(10)},
    "confusion_matrix": cm.tolist(),
    "split_counts": counts,
    "n_train": len(x_train),
    "n_val": len(x_val),
    "n_test": len(x_test_c),
    "misclassified": int((pred != y_test).sum()),
    "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
}
json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)

# ------------------------------------------------------------- figures
h = hist.history
ep = np.arange(1, epochs_run + 1)
plt.figure(figsize=(9, 3.6))
plt.plot(ep - 1, h["loss"], label="Training Loss")
plt.plot(ep - 1, h["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/loss_curve.png", dpi=200); plt.close()

plt.figure(figsize=(9, 3.6))
plt.plot(ep - 1, h["accuracy"], label="Training Accuracy")
plt.plot(ep - 1, h["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/accuracy_curve.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6.2, 5.6))
ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(ax=ax, cmap="Blues", colorbar=True)
ax.set_title("Handwritten Digit Recognition - Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrix.png", dpi=200); plt.close()

# sample dataset grid (one per class)
fig, axs = plt.subplots(2, 10, figsize=(11, 2.9))
rng = np.random.default_rng(SEED)
for c in range(10):
    idx = np.where(y_test == c)[0]
    for row in range(2):
        i = rng.choice(idx)
        axs[row, c].imshow(x_test[i], cmap="gray"); axs[row, c].axis("off")
        if row == 0: axs[row, c].set_title(str(c))
plt.suptitle("Sample MNIST images", y=1.02); plt.tight_layout()
plt.savefig(f"{OUT}/dataset_samples.png", dpi=200, bbox_inches="tight"); plt.close()

# sample predictions
sel = rng.choice(len(x_test), 20, replace=False)
fig, axs = plt.subplots(2, 10, figsize=(12, 3.2))
for a, i in zip(axs.ravel(), sel):
    a.imshow(x_test[i], cmap="gray"); a.axis("off")
    ok = pred[i] == y_test[i]
    a.set_title(f"T:{y_test[i]} P:{pred[i]}", fontsize=9, color="green" if ok else "red")
plt.tight_layout(); plt.savefig(f"{OUT}/sample_predictions.png", dpi=200); plt.close()

# misclassified
wrong = np.where(pred != y_test)[0]
show = wrong[:20]
fig, axs = plt.subplots(2, 10, figsize=(12, 3.2))
for a in axs.ravel(): a.axis("off")
for a, i in zip(axs.ravel(), show):
    a.imshow(x_test[i], cmap="gray")
    a.set_title(f"T:{y_test[i]} P:{pred[i]}", fontsize=9, color="red")
plt.tight_layout(); plt.savefig(f"{OUT}/misclassified.png", dpi=200); plt.close()
print("DONE")
