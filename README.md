# Handwritten Digit Recognition using CNN

A three-block CNN trained on MNIST, with a Streamlit demo app.

## Files
- `train.py` - loads MNIST, trains the CNN, evaluates it, saves `outputs/digit_cnn.keras`, `outputs/metrics.json` and all graphs.
- `app.py` - Streamlit user interface (upload image / draw digit / test-set sample).
- `preprocess.py` - converts any digit image to the 28x28 white-on-black MNIST format.
- `data/mnist.npz` - MNIST arrays (`x_train, y_train, x_test, y_test`).
- `outputs/` - trained model, metrics, confusion matrix, curves, sample predictions.

## Run
```bash
pip install -r requirements.txt
python train.py            # optional: a trained model is already in outputs/
streamlit run app.py
```

If `data/mnist.npz` is missing, create it from the standard MNIST files, or replace the
loading lines in `train.py` with `keras.datasets.mnist.load_data()` (needs internet).

## Results (10,000 test images)
Accuracy 99.14%, macro precision 99.14%, macro recall 99.14%, macro F1 99.14%, test loss 0.0239.
