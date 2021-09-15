## Machine Learning Collective Variables (mlcvs)

Python package to train the CVs, using the machine learning library Pytorch. 

To install the requirements:

```
pip install -r requirements.txt
```

Note: pytorch version 1.5 is specified in the requirements, in order to match the [instructions](https://colab.research.google.com/github/luigibonati/data-driven-CVs/blob/master/code/Tutorial%20-%20DeepLDA%20training.ipynb#scrollTo=F7qSDdBGn8Vv) to compile the LibTorch C++ interface and use the resulting models in PLUMED. However, the code can be used also with a more recent version of Pytorch, provided that the version of LibTorch matches it.

Look at the `tutorial` folder for how to use the code to train the CVs.
