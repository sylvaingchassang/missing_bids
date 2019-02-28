##### Code for 
#### ''Data-Driven Regulation: Theory and Application to Missing Bids'' 
##### (Chassang, Kawai, Nakabayashi and Ortner)

tested under Python 3.6.6, and Ubuntu Linux 16.04/18.04

if you have suggestions for improvement, please contact us  
if you want to contribute, follow [PEP8](https://www.python.org/dev/peps/pep-0008/) standards


**Instructions**
1. to clone the repo:

    > git clone git@github.com:sylvaingchassang/missing_bids.git

1. to install requirements, run
    > pip install -r requirements.txt

    from inside the `missing_bids/` folder

1. to run tests and check coverage, run
    > sh missing_bids/test_coverage.sh
    
    from outside the `missing_bids/` folder
    
1. to play with an example notebook illustrating key functionality, run
    > jupyter notebook example_tsuchiura.ipynb
    
    from inside the `missing_bids/notebooks/` folder
    
    make sure the folder containing `missing_bids/` is included in your `PYTHONPATH`

1. to generate figures included in the paper, run
    > bash missing_bids/scripts/generate_figures.sh 
    
    generating figures requires 16GB RAM;  
    this is caused by memory leaks while bootstrapping the national level data   
    suggestions for improvements welcome
