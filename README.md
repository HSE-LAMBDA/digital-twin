# Digital Twin

This is the official repository for "Performance Modeling of Data Storage Systems using Generative Models" paper

## Abstract
High-precision modeling of systems is one of the main areas of industrial data
analysis. Models of systems, their digital twins, are used to predict their behavior under various
conditions. We have developed several models of a storage system using machine
learning-based generative models. The system consists of several components: hard disk drive
(HDD) and solid-state drive (SSD) storage pools with different RAID schemes and cache. Each
storage component is represented by a probabilistic model that describes the probability
distribution of the component performance in terms of IOPS and latency, depending on their
configuration and external data load parameters. The results of the experiments demonstrate the
errors of 4–10 % for IOPS and 3–16 % for latency predictions depending on the components and
models of the system. The predictions show up to 0.99 Pearson correlation with Little’s law,
which can be used for unsupervised reliability checks of the models. In addition, we present
novel data sets that can be used for benchmarking regression algorithms, conditional generative
models, and uncertainty estimation methods in machine learning.

## Citation

```
@article{al2023performance,
  title={Performance Modeling of Data Storage Systems using Generative Models},
  author={Al-Maeeni, Abdalaziz Rashid and Temirkhanov, Aziz and Ryzhikov, Artem and Hushchyn, Mikhail},
  journal={arXiv preprint arXiv:2307.02073},
  year={2023}
}
```

## Instalation

```
# clone the project to your local machine
git clone https://github.com/HSE-LAMBDA/digital-twin.git

# install package
cd digital-twin && pip install .
```

## Project structure
```
  .
    ├── digital_twin                                        #  Directory with a library source code
    │   ├── models                                          #
    |   |   ├── density estimation                          #
    |   |   |    ├── gmm.py                                 #
    |   |   |    ├── knn.py                                 #
    |   |   |    ├── regressor.py                           #
    |   |   |    ├── grouper.py                             #
    |   |   ├── norm_flow                                   #  Directory for normalized-flow-based digital twin model
    |   |   |    ├── utils                                  #  
    |   |   |    ├── model.py                               #  
    |   ├── performance metrics                             #
    |   |   ├── fd.py                                       #  Frechet Distance metric
    |   |   ├── misc.py                                     # 
    |   |   ├── mmd.py                                      #  Maximum Mean Discrepancy metric 
    |   ├── visualization                                   #
    |   |   ├── plot.py                                     #  
    |   ├── data.py                                         #
    ├── model_checkpoints                                   #
    ├── results                                             #  Directory with experiments results
    ├── scripts                                             #  Scripts to reproduce experiments
    |   ├── calculate_summary_stats.py                      #  This script calculates metrics and plot prediction plots based on prediction csv files
    |   ├── denesity_estimation_experiment.py               #
    |   ├── generate_train_test_split.py                    #
    |   ├── knn_experiment.py                               # 
    |   ├── nf_experiment.py                                #  This script is used for experiment with normalized flow
```
