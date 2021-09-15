## Data

Each folder contains the input data for the different systems presented in the manuscript.

```
.
├── ala2-multi
├── ala2-psi
├── chignolin
└── silicon
```

Each folder is structured in the same way, and contains (at least) three subfolders following the Deep-TICA procedure presented in the manuscript:

1. Exploratory simulation using trial CVs or multicanonical ensemble 
2. Train NN to obtain the Deep-TICA CVs (with python, see `../tutorial` folder)
3. New simulation enhancing the sampling of the leading Deep-TICA CV 
