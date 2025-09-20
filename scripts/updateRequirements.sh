#!/usr/bin/env bash
pip-compile --strip-extras -q -o requirements.txt pyproject.toml
pip-compile --strip-extras --extra=dev -q -o requirements-dev.txt pyproject.toml
pip-compile --strip-extras --extra=test -q -o requirements-test.txt pyproject.toml
pip-compile --strip-extras --extra=demo -q -o requirements-demo.txt pyproject.toml
