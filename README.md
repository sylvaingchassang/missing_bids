##### Code for 
#### ''Data-Driven Regulation: Theory and Application to Missing Bids'' 
##### (Chassang, Kawai, Nakabayashi and Ortner)

![](https://travis-ci.com/sylvaingchassang/missing_bids.svg?branch=master) 

tested under Python 3.8, and Ubuntu Linux 16.04/18.04, @travis-ci
make sure to install python3.8-dev

if you have suggestions for improvement, please contact us  
if you want to contribute, follow [PEP8](https://www.python.org/dev/peps/pep-0008/) standards


**Instructions**
1. to clone the repo:

    `git clone git@github.com:sylvaingchassang/missing_bids.git`

1. to install requirements, run

    `pip install -r requirements.txt`

    from inside the `missing_bids/` folder
    
1. to run tests and check coverage, run
    
    `pytest --cov=.`
    
    from within the `missing_bids/mb_api/` folder
    
1. to generate figures included in the paper, run
    `bash missing_bids/scripts/round3/generate_figures.sh`
    
    generating figures requires 16GB RAM;  
    suggestions for improvements welcome
