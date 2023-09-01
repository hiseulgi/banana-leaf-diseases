#!/bin/bash

docker run --rm -v "$PWD"/temp:/src/temp banana-leaf-diseases python main.py "$@"