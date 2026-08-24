SHELL := /bin/sh
PYTHON ?= python3
ROOT := $(CURDIR)/..
export PYTHONPATH := $(CURDIR)/src

ifneq (,$(wildcard $(ROOT)/.env))
include $(ROOT)/.env
export
endif

.PHONY: setup test fetch build smoke run paper clean

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -q -e .

test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

fetch:
	.venv/bin/python -m popstats.pipeline fetch --state ca --file both
	.venv/bin/python -m popstats.pipeline fetch --state ny --file both

build:
	.venv/bin/python -m popstats.pipeline build --state ca
	.venv/bin/python -m popstats.pipeline build --state ny

smoke:
	.venv/bin/python -m popstats.pipeline smoke

run:
	.venv/bin/python -m popstats.pipeline run

paper:
	.venv/bin/python -m popstats.pipeline paper

clean:
	.venv/bin/python -m popstats.pipeline clean
