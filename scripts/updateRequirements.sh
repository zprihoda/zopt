#!/usr/bin/env bash
pip-compile --strip-extra -o requirements.txt pyproject.toml
pip-compile --extra=dev -o requirements-dev.txt pyproject.toml
pip-compile --extra=test -o requirements-test.txt pyproject.toml
pip-compile --extra=demo -o requirements-demo.txt pyproject.toml
