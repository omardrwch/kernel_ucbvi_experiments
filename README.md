# Kernel-Based Reinforcement Learning: A Finite-Time Analysis

Implementing the `KernelUCBVI` algorithm from the paper. The algorithm and the baselines are implemented in the folder `algorithms/`. The folder `config/` contains the parameters defining the experiments.

* Requirements:
    * Python 3.7
    * [`rlberry`](https://github.com/rlberry-py/rlberry) version 0.1
    * pyyaml

* Create and activate conda virtual environment (optional)

```bash
$ conda create -n kernel_ucbvi_env python=3.7
$ conda activate kernel_ucbvi_env
```

* Install requirements

```bash
$ pip install 'rlberry[full]==0.1'
$ pip install pyyaml
```


* Run `Kernel-UCBVI` and `AdaptiveQL` experiments:

```
$ python run.py config/experiments/twinrooms_exp.yaml --n_fit=8
```

* Run `UCBVI` and `OptQL` experiments:

```
$ python run.py config/experiments/twinrooms_exp_unif_discr.yaml --n_fit=8
```

* Generate the plots:

```
$ python plot.py
```
