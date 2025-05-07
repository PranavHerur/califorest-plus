# CaliForest

**Cali**brated Random **Forest**

This Python package implements the CaliForest algorithm presented in [ACM CHIL 2020](https://www.chilconference.org/).

You can use CaliForest almost the same way you used RandomForest i.e. you can just replace RandomForest with CaliForest.
The only difference would be that its predicted scores will be better calibrated than the regular RandomForest output, while maintaining the original predictive performance.
For more details, please see "CaliForest: Calibrated Random Forest for Health Data" in ACM Conference on Health, Inference, and Learning 2020. 

![](analysis/hastie-results.png)

## Installing

Installing from the source:

```
$ git clone git@github.com:yubin-park/califorest.git
$ cd califorest
$ python setup.py develop
```

## Example Code

Training + Prediction:

```python
from califorest import CaliForest

model = CaliForest(n_estimators=100,
                    max_depth=5,
                    min_samples_split=3,
                    min_samples_leaf=1,
                    ctype="isotonic")

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:,1]
```

Calibration metrics:

```python
from califorest import metrics as em

score_auc = roc_auc_score(y_test, y_pred)
score_hl = em.hosmer_lemeshow(y_test, y_pred)
score_sh = em.spiegelhalter(y_test, y_pred)
score_b, score_bs = em.scaled_Brier(y_test, y_pred)
rel_small, rel_large = em.reliability(y_test, y_pred)
```

## Reproduce Original CaliForest Experiment
1. download the MIMIC-III dataset from https://physionet.org/content/mimiciii/1.4/
2. process it using https://github.com/MIT-LCP/mimic-code/
3. process that using https://github.com/MLforHealth/MIMIC_Extract
4. then put that output of that `all_hourly_data.h5` into a folder named directory at the top level of this project

### Then run
1. `pip install -e .`
2. `cd analysis`
3. `python califorest_original_experiment.py`
4. You'll see new folder named `califorest_original_results` containg a csv and a image of the results

## License

MIT License

## Reference

Y. Park and J. C. Ho. 2020. **CaliForest: Calibrated Random Forest for Health Data**. *ACM Conference on Health, Inference, and Learning (2020)*




