#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

# Plane data

if [ ! -f plane-data.csv ]; then
    wget http://stat-computing.org/dataexpo/2009/plane-data.csv
fi

if [ ! -f 2008.csv ]; then
    if [ ! -f 2008.csv.bz2 ]; then
        wget http://stat-computing.org/dataexpo/2009/2008.csv.bz2
    fi
    bzip2 -d 2008.csv.bz2
fi

if [ ! -f plane_data.npy ]; then
    python parse_plane_data.py
fi
