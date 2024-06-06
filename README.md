# Deep learning the slow modes for rare events sampling
#### Luigi Bonati, GiovanniMaria Piccini, and Michele Parrinello, _arXiv preprint_ arXiv:2107.03943 (2021).

[![PNAS](https://img.shields.io/badge/PNAS-2021_118_(44)-blue)](https://doi.org/10.1073/pnas.2113533118)
[![arXiv](https://img.shields.io/badge/arXiv-2107.03943-critical)](https://arxiv.org/abs/2107.03943)
[![MaterialsCloud](https://img.shields.io/badge/MaterialsCloud-10.24435-lightgrey)](https://doi.org/10.24435/materialscloud:3g-9x)
[![plumID:21.039](https://www.plumed-nest.org/eggs/21/039/badge.svg)](https://www.plumed-nest.org/eggs/21/039/)

> [!IMPORTANT]
> This repository is kept as supporting material for the manuscript, but it is no longer updated. Check out the [mlcolvar](https://mlcolvar.readthedocs.io)  library for data-driven CVs, where you can find up-to-date tutorials and examples.
> 
> [<img src="https://raw.githubusercontent.com/luigibonati/mlcolvar/main/docs/images/logo_name_black_big.png" width="200" />](https://mlcolvar.readthedocs.io)


This repository contains input data and code related to the manuscript:

* `data` --> input files for the simulations and the CVs training
* `mlcvs` --> python package to train the Deep-TICA CVs
* `plumed-libtorch-interface` --> interface to load Pytorch models in PLUMED2 for enhanced sampling
* `tutorial` --> jupyter notebook with tutorial to train the CVs

Due to size limits the outputs trajectories of Chignolin and Silicon simulations are deposited in the [Materials Cloud](https://doi.org/10.24435/materialscloud:3g-9x) repository.
