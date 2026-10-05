import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

def predict_churn(encoded_unscaled_features, artifact_dir):
    root = Path(artifact_dir)
    meta = json.loads((root / 'metadata.json').read_text())
    expected = meta['feature_columns']
    if set(encoded_unscaled_features.columns) != set(expected):
        raise ValueError('Provide exactly the documented encoded, unscaled features, without Churn.')
    x = encoded_unscaled_features[expected].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Input contains missing or non-finite values.')
    with np.load(root / 'scaler_parameters.npz') as params:
        x = ((x - params['mean']) / params['scale']).astype('float32')
    model = tf.keras.models.load_model(root / 'ann_final.keras', compile=False)
    p = model.predict(x, verbose=0).ravel()
    return pd.DataFrame({'churn_probability': p,
        'predicted_churn': (p >= meta['threshold']).astype(int)},
        index=encoded_unscaled_features.index)
