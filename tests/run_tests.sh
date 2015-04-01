#!/bin/bash

pushd $(dirname $0) &> /dev/null
python2 -m unittest discover
popd &> /dev/null
