#!/usr/bin/env bash

python -m tsuchiura_figures
pkill python
python -m industry_figures
pkill python
python -m national_sample_high_low_bids_figures
pkill python
python -m individual_firms
pkill python