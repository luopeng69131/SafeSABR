#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
c++ -O4 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) corerl.cc -o libcorerl$(python3-config --extension-suffix)
cp libcorerl*.so ../sim_env/
echo 'Done'
