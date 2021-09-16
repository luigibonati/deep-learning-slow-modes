# Deep learning the slow modes for rare events sampling
#### Luigi Bonati, GiovanniMaria Piccini, and Michele Parrinello, _arXiv preprint_ arXiv:2107.03943 (2021).

[![arXiv](https://img.shields.io/badge/arXiv-2107.03943-critical)](https://arxiv.org/abs/2107.03943)
[![MaterialsCloud](https://img.shields.io/badge/MaterialsCloud-10.24435-lightgrey)](https://doi.org/10.24435/materialscloud:3g-9x)
[![plumID:21.039](https://www.plumed-nest.org/eggs/21/039/badge.svg)](https://www.plumed-nest.org/eggs/21/039/)

This repository contains input data and code related to the manuscript:

* `data` --> input files for the simulations and the CVs training
* `mlcvs` --> python package to train the Deep-TICA CVs
* `plumed-libtorch-interface` --> interface to load Pytorch models in PLUMED2 for enhanced sampling
* `tutorial` --> jupyter notebook with tutorial to train the CVs

Due to size limits the outputs trajectories of Chignolin and Silicon simulations are deposited in the [Materials Cloud](https://doi.org/10.24435/materialscloud:3g-9x) repository.
