#!/usr/bin/env bash
ROOT_DIR=$(pwd)
echo "ROOT_DIR=${ROOT_DIR}"

for EXTENSION_DIR in ops/*
do
    if [ -d "${EXTENSION_DIR}" ] ; then
        echo ${EXTENSION_DIR}
        cd ${EXTENSION_DIR}
        if [ -d "build" ]; then
            rm -r build
        fi
        python setup.py build_ext --inplace
    fi
    cd ${ROOT_DIR}
done