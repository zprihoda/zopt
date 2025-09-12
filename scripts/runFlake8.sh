#!/usr/bin/env bash
flake8 . --count --exit-zero  --statistics --config=pyproject.toml --exclude ".*"
