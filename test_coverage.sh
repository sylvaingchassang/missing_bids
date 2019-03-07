#!/usr/bin/env bash

coverage run --source=missing_bids/ --omit=*/tests/*,*__init__.py,*/scripts/* \
-m py.test missing_bids/tests/

coverage report -m
