#!/usr/bin/env bash

python -m national_sample
pkill python
python -m national_sample_high_low_bids_figures
pkill python
python -m national_industry_figures
pkill python
python -m individual_national_firms
pkill python
python -m city_level_individual_firms
pkill python
python -m city_level_high_low
pkill python
python -m city_level_all
pkill python