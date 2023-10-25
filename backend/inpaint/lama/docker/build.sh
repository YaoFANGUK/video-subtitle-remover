#!/bin/bash

BASEDIR="$(dirname $0)"

docker build -t windj007/lama -f "$BASEDIR/Dockerfile" "$BASEDIR"
