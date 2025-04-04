# Performance Modeling of Data Storage Systems Using Generative Models

This is the official repository for paper "Performance Modeling of Data Storage Systems using Generative Models"  in IEEE Access [[journal]](https://doi.org/10.1109/ACCESS.2025.3552409) [[arxiv]](https://arxiv.org/abs/2307.02073).

## Abstract
High-precision systems modeling is one of the main areas of industrial data analysis. Models of systems, their digital twins, are used to predict their behavior under various conditions. In this study, we developed several models of a storage system using machine learning-based generative models to predict performance metrics such as IOPS and latency. The models achieve prediction errors ranging from 4%–10% for IOPS and 3%–16% for latency and demonstrate high correlation (up to 0.99) with observed data. By leveraging Little’s law for validation, these models provide reliable performance estimates. Our results outperform conventional regression methods, offering a vendor-agnostic approach for simulating data storage system behavior. These findings have significant applications for predictive maintenance, performance optimization, and uncertainty estimation in storage system design.

## Citation
If you use the scripts or datasets in a scientific publication, we would appreciate citations to the paper:

```
A. R. Al-Maeeni, A. Temirkhanov, A. Ryzhikov and M. Hushchyn, "Performance Modeling of Data Storage Systems Using Generative Models," in IEEE Access, vol. 13, pp. 49643-49658, 2025, doi: 10.1109/ACCESS.2025.3552409
```

or using BibTeX:

```
@ARTICLE{10930879,
  author={Al-Maeeni, Abdalaziz R. and Temirkhanov, Aziz and Ryzhikov, Artem and Hushchyn, Mikhail},
  journal={IEEE Access}, 
  title={Performance Modeling of Data Storage Systems Using Generative Models}, 
  year={2025},
  volume={13},
  number={},
  pages={49643-49658},
  doi={10.1109/ACCESS.2025.3552409}}
```
## Datasets

All datasets used in the paper are in [dataset.tar.gz](https://github.com/HSE-LAMBDA/digital-twin/blob/master/dataset.tar.gz) archive with the following structure:

```bash
dataset
├── cache
│   ├── cache.csv
│   ├── test_cache.csv
│   └── train_cache.csv
└── pools
    ├── hdd_sequential.csv
    ├── ssd_random.csv
    ├── ssd_sequential.csv
    ├── test_hdd_sequential.csv
    ├── test_ssd_random.csv
    ├── test_ssd_sequential.csv
    ├── train_hdd_sequential.csv
    ├── train_ssd_random.csv
    └── train_ssd_sequential.csv
```

## Instalation

```
# clone the project to your local machine
git clone https://github.com/HSE-LAMBDA/digital-twin.git

# install package
cd digital-twin && pip install -e .
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
