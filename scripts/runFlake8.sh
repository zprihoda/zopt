#!/usr/bin/env bash
flake8 . --count --exit-zero  --statistics --config=setup.cfg --exclude ".*"
