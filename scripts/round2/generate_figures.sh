#!/usr/bin/env bash

python -m national_sample
pkill python
python -m national_sample_high_low_bids_figures
pkill python