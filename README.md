# MLBugDetection

Instalation: 

```bash
pip install mlbugdetection
```

# Functions


1. Monotonic:
    ```py
    from mlbugdetection.monotonic import monotonicity_mse, check_monotonicity
    ```

    Usage:
    ```py
    check_monotonicity(model, sample, feature, start, stop, steps=100)
    ```

2. Critical Values:
    ```py
    from mlbugdetection.critical_values import find_critical_values, highest_and_lowest_indexes
    ```
    
    Usage:
    For 1 sample
    ```py
    find_critical_values(model, sample, feature, start, stop, step=100)
    ```

    For more than 1 sample:
    ```py
    find_several_critical_values(model, samples, feature, start, stop, steps=100, bins=15, keep_n=5)
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
    from mlbugdetection.sanity import sanity_check
    ```

    Usage:

    ```py
    sanity_check(model, samples, target)
    ```

    If test don't pass, check the indexes
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