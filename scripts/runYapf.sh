#!/usr/bin/env bash
yapf . --style=pyproject.toml --in-place --recursive -e ".*"
