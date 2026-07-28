# InvLT: Monotonic Logit Transformation for Post-hoc Calibration


## NOTE ON THE IMAGENET LOGITS

The ImageNet ResNet-152 logits are ~191 MB per seed, too large to host here,
so `logits/` is empty. The CIFAR-10 and CIFAR-100 logits are included, so
those four tasks run as-is. 

## QUICKSTART

```bash
pip install torch numpy scipy scikit-learn
# python rerun_all.py   # all datasets
python rerun_all.py --only-dataset cifar10
python rerun_all.py --only-dataset cifar100
```

Runs all 12 methods x 5 seeds on both CIFAR validation sizes (500 and 5000),
using the cached logits in `models/`. 

Quick sanity check instead (~2 minutes):

```bash
python rerun_all.py --only-dataset cifar10 --max-seeds 1 --skip-methods umnn_calibration
```

## THE SEVEN TASKS

```
CIFAR-10   (ResNet-50)    val 500, 5000
CIFAR-100  (ResNet-50)    val 500, 5000
ImageNet   (ResNet-152)   val 500, 5000, 25000
```

"val N" = the calibrator is fitted on the first N validation samples. The test
set is unchanged. All results are means over seeds 0-4.

## OUR METHOD

```
Calibrator/monotonic_logits_transformation.py
    class MonotonicLogitsTransformation
```

An element-wise scalar MLP f applied to every logit, with an inverse g trained
jointly so that g(f(x)) ~ x. That reconstruction term softly enforces
monotonicity, so the transformation preserves the predicted class and leaves
accuracy untouched.

```
L = CE(softmax(f(z)), y) + lam * (1/K) * sum_k (g(f(x_k)) - x_k)^2
```

Three hyperparameters were searched (hidden_dims x activation x loss = 24
configurations per task). The winners are in evaluate.py, applied
automatically:

```
task                  activation  hidden_dims    loss
CIFAR-10   val 500    tanh        (8, 8)         ce
CIFAR-10   val 5000   tanh        (16, 16)       ce
CIFAR-100  val 500    tanh        (16, 16, 16)   ce
CIFAR-100  val 5000   tanh        (24, 24)       ce
ImageNet   val 500    relu        (8, 8)         ce
ImageNet   val 5000   relu        (8, 8)         ce
ImageNet   val 25000  relu        (8, 8)         ce
```


## RUNNING ONE THING

```bash
python evaluate.py --logits-path unused \
  --logits-file-pattern './models/cifar100/resnet50/cifar100_resnet50_b128_e350_seed{seed}_logits.pt' \
  --dataset cifar100 --model resnet50 --method-name logits_transformation \
  --val-size 5000 --seed 0 --output ./results/cifar100_val5000/resnet50_s0.txt

python average_seeds.py --inputs results/cifar100_val5000/resnet50_s{0,1,2,3,4}.txt \
                        --output results/cifar100_val5000/resnet50.txt
```

`--method-name` accepts: no_calibration, temperature_scaling, histogram_binning,
isotonic_regression, vector_scaling, matrix_scaling, dirichlet_calibration,
pts, ets, spline_calibration, umnn_calibration, logits_transformation.

## LAYOUT

```
Calibrator/     our method + all baselines (umnn/ is the UMNN integrand)
Metrics/        ECE, AdaECE, classwise ECE, KDE-ECE, max-ECE
Net/            network architectures (only needed to retrain from scratch)
Data/           dataset loaders (only needed to retrain from scratch)
models/         cached CIFAR-10/100 logits, seeds 0-4 (included, 131 MB)
logits/         ImageNet logits -- empty, see note above
results/        output; one file per task, plus per-seed files
                (cifar100_val5000/ is included as a reference result)
evaluate.py     one method, one seed, one task
rerun_all.py    all methods, all seeds, all seven tasks
average_seeds.py   collapses per-seed files into mean (std)
pretrain.py     trains the base networks (not needed to reproduce tables)
extract_imagenet_logits.py   regenerates the ImageNet logits
```

## NOTES

- Baselines run at their published defaults.
- Results are 5-seed means. Individual seeds vary noticeably at val 500, where
  only 500 samples are available to fit on.
- ECE is a binned statistic and is sensitive to the exact stopping point of
  training; NLL is far more stable. Small differences between a rerun and the
  printed tables are expected, especially on ImageNet.

