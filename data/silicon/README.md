## Alanine dipeptide - multicanonical simulation

```
.
├── 0_unbiased_dlda		--> unbiased simulation at 1700K of liquid and solid
├── 1_opes_dlda			--> exploratory simulation
├── 2_training_cvs		--> train Deep-TICA CVs
├── 3_opes_dlda_dtica		--> enhance sampling of Deep-TICA C
├				
└── StructureFactor.miller.cpp	--> Code for the calculation of the 3D S(k)
``` 

The Deep-LDA CV has been constructed using the code released in [Google Colab](https://colab.research.google.com/github/luigibonati/data-driven-CVs/blob/master/code/Tutorial%20-%20DeepLDA%20training.ipynb), see also the repository of [Deep-LDA](https://github.com/luigibonati/data-driven-CVs).
