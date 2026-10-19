#!/bin/sh
# Compiles SEFR.h on the host and compares it with a NumPy implementation.
# Needs a C++11 compiler and Python with numpy and scikit-learn.
set -e
cd "$(dirname "$0")"
c++ -std=c++11 -O2 -Wall -Wextra -Wno-unused-result -o test_host test_host.cpp
python3 reference.py ./test_host
