##### Code for 
#### ''Data-Driven Regulation: Theory and Application to Missing Bids'' 
##### (Chassang, Kawai, Nakabayashi and Ortner)

![](https://travis-ci.com/sylvaingchassang/missing_bids.svg?branch=master) 

tested under Python 3.7.4, and Ubuntu Linux 16.04/18.04, @travis-ci

hoping 3.8 works, make sure to run sudo apt-get install python3.8-dev

if you have suggestions for improvement, please contact us  
if you want to contribute, follow [PEP8](https://www.python.org/dev/peps/pep-0008/) standards


**Instructions**
1. to clone the repo:

    `git clone git@github.com:sylvaingchassang/missing_bids.git`

1. to install requirements, run

    `pip install -r requirements.txt`

    from inside the `missing_bids/` folder
    
    note that the `cvxpy` package is a little finicky; 
    it works smoothly under python 3.7.4 but not necessarily under later versions

1. to run tests and check coverage, run
    
    `sh missing_bids/test_coverage.sh`
    
    from outside the `missing_bids/` folder
    
1. to play with an example notebook illustrating key functionality, run
    `jupyter notebook example_tsuchiura.ipynb`
    
    from inside the `missing_bids/notebooks/` folder
    
    make sure the folder containing `missing_bids/` is included in your `PYTHONPATH`

1. to generate figures included in the paper, run
    `bash missing_bids/scripts/round2/generate_figures.sh`
    
    generating figures requires 16GB RAM;  
    suggestions for improvements welcome
