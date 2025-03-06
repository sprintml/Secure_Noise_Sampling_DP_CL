# Implementation of Dice Ensemble Approximation

## Setup
To install the required python packages, you can use:
``` 
pip install numpy
pip install scipy
pip install matplotlib
```

## Using the script
The primary item of interest in `dice_ensemble_approximation.py` is an implementation of Algorithm 4 in the manuscript, dice ensemble approximation. We provide examples for running the algorithm on discrete gaussian distributions with varying parameters, and a simple toy distribution, at the bottom of the script. These can be executed using
```python dice_ensemble_approximation.py```
and interactive mode can be used to inspect them further. Other distributions besides those included in the script can be approximated by representing probability mass functions as python dictionaries and initializing the DiceEnsemble class as shown in the "constructing dice ensembles" section near the bottom of the script. We also include a method for generating a plot which shows required table size for the guarantees proven in the paper to hold at varying parameters of the discrete gaussian. Uncomment the last line of the script to see the plot.