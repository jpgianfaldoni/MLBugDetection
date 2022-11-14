# MLBugDetection

### Machine learning explainability and unexpectated behaviors detection

## Overview

Most machine learning explainability packages requires both trained models and the training data to create Explainer objects that explain the model's behavior. This package allows ceteris paribus analysis of features using only the trained model and one or more input samples.

## Documentation

https://jpgianfaldoni.github.io/MLBugDetection/

Instalation: 

```bash
pip install mlbugdetection
```

## Functions


1. Monotonic:
    ```py
    from mlbugdetection.monotonic import check_monotonicity_single_sample, check_monotonicity_multiple_samples
    ```

    Usage:
    For 1 sample
    ```py
    check_monotonicity_single_sample(model, sample, feature, start, stop, steps=100)
    ```
    
    For more than 1 sample:
    ```py
    check_monotonicity_multiple_samples(model, sample, feature, start, stop, steps=100) 
    ```
    

2. Critical Values:
    ```py
    from mlbugdetection.critical_values import find_critical_values, find_several_critical_values
    ```
    
    Usage:
    For 1 sample
    ```py
    find_critical_values(model, sample, feature, start, stop, step=100)
    ```

    For more than 1 sample:
    ```py
    find_several_critical_values(model, samples, feature, start, stop, steps=100, bins=15, keep_n=5, log=False)
    ```


3. Calibration:
    ```py
    from mlbugdetection.calibration import calibration_check
    ```
    
    Usage:
    ```py
    calibration_check(model, samples, target)
    ```

4. Sanity:
    ```py
    from mlbugdetection.sanity import sanity_check, sanity_check_with_indexes
    ```

    Usage:

    ```py
    sanity_check(model, samples, target)
    ```

    If test not pass, check the indexes
    ```py
    sanity_check_with_indexes(model, samples, target)
    ```


---

## Virtual Environment with Jupyter Notebook

```bash
python3 -m virtualenv venv 
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
