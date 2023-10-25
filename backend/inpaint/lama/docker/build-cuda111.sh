#!/bin/bash

BASEDIR="$(dirname $0)"

docker build -t windj007/lama:cuda111 -f "$BASEDIR/Dockerfile-cuda111" "$BASEDIR"
