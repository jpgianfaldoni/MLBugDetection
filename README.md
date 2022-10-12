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
    ```py
    find_critical_values(model, sample, feature, start, stop, step=100)
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


---

## Virtual Environment with Jupyter Notebook

```bash
python3 -m virtualenv venv 
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```