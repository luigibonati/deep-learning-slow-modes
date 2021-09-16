## PLUMED - Pytorch interface

In order to use the Deep-TICA CVs to enhance the sampling, we need to load the NN models inside PLUMED.

This folder contains a Plumed-libtorch interface, as developed for the [Deep-LDA](https://github.com/luigibonati/data-driven-CVs) NN CVs.

**(1) Download Libtorch**

Prebuilt binaries can be downloaded from [Pytorch](https://pytorch.org/) website. Both binaries (cxx11 ABI and pre-cxx11) can be used. Note that the versions of Pytorch and LibTorch should match in order to load correctly the serialized model. The following instructions are related to [LibTorch 1.4](http://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.4.0%2Bcpu.zip), which work with the model specified in the `mlcvs` requirements.

**(2) Add PytorchModel.cpp to PLUMED**

You need to add this the `PytorchModel.cpp` in the source directory of PLUMED2 (e.g. `plumed2/src/function/`) and, after configuring PLUMED with the Libtorch library, recompile PLUMED. 
Alternatively, this interface can be also loaded in runtime using the LOAD command in the PLUMED input file: 
```
LOAD FILE=PytorchModel.cpp
```
Note that also in this second case you need to configure PLUMED to include the Libtorch library (see below), so I suggest to recompile PLUMED with the .cpp file in it, so that you can immediately detect if the linking was succesful.

**(3) Configure PLUMED**

If `$LIBTORCH` contains the location of the downloaded binaries, we can configure PLUMED in the following way:

```
./configure  --enable-rpath \
             --disable-external-lapack --disable-external-blas \
             CXXFLAGS="-O3 -D_GLIBCXX_USE_CXX11_ABI=0" \
             CPPFLAGS="-I${LIBTORCH}/include/torch/csrc/api/include/ -I${LIBTORCH}/include/ -I${LIBTORCH}/include/torch" \
             LDFLAGS="-L${LIBTORCH}/lib -ltorch -lc10 -Wl,-rpath,${LIBTORCH}/lib"
```

Notes:
- This command is valid for the pre-cxx11 ABI version. If you downloaded the cxx11 ABI one the corresponding option should be enabled in the configure: `CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"`
- When using different versions of Libtorch the name of the libraries to be linked might be different.


**(4) Load the model in the input file**

In the PLUMED input file one should specify the model and the arguments. The interface detects the number of outputs and create a component for each of them, which can be accessed as cv.node-0, cv.node-1, ... 
```
cv: PYTORCH_MODEL MODEL=model.pt ARG=d1,d2,...,dN
```
