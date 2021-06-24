##### Code for 
#### ''Robust Screens for Non-Competitive Bidding in Procurement Auctions'' 
##### (Chassang, Kawai, Nakabayashi and Ortner)

![](https://travis-ci.com/sylvaingchassang/missing_bids.svg?branch=master) 

tested under Python 3.8, and Ubuntu Linux 16.04/18.04/20.04 
make sure to install `python3.8-dev`

if you have suggestions for improvement, please contact us  
if you want to contribute, follow [PEP8](https://www.python.org/dev/peps/pep-0008/) standards


**Instructions**
1. to clone the repo:

    `git clone git@github.com:sylvaingchassang/missing_bids.git`

1. to install requirements, run

    `pip install -r requirements.txt`

    from inside the `missing_bids/` folder
    
    note that the `cvxpy` package is a little finicky; 
    it works smoothly under python 3.8

1. to run tests and check coverage, run
    
    `pytest --cov=.`
    
    from within the `missing_bids/` folder
    
    make sure the folder containing `missing_bids/` is included in your `PYTHONPATH`

1. to generate figures included in the paper

download the [data](https://www.dropbox.com/s/kigyfge4ubc8er3/data_missing_bids.zip?dl=0) and extract it to a folder `missing_bids_data` with the following structure:
- `missing_bids_data/`

        - bc_collusion.csv
        - other `csv` files

from within the folder `missing_bids/scripts/` create a file named `script_config.py` containing a single line:

```path_data = 'path/to/missing_bids_data'```
where `path/to` is the absolute path to the `missing_bids_data` folder you created.


then, from within  `missing_bids/scripts/` run 

    `bash generate_figures.sh`
    
figures will be outputed to the `missing_bids_data` folder
    
generating figures requires at least 16GB RAM;  
suggestions for improvements welcome

1. Computation parameters are specified in the `missing_bids/scripts/figures_import_helper.py` file

    - figures used in the paper are generated using parameters
      NUM_POINTS = 1000, NUM_EVAL = 200, SEEDS = [0, 1]
      Computations take  roughly one week on an 8 core (dual cpu) intel i7-3770 clocked at 3.4 GHz, with 32G of RAM
    - very similar figures can be generated within a third of the time by using parameters 
      NUM_POINTS = 1000, NUM_EVAL = 100, SEEDS = [0]
    - to test that everything is working, it is useful to run computations with parameters 
      NUM_POINTS = 500, NUM_EVAL = 2, SEEDS = [0]
      under that configuration, computations typically take under one hour.

