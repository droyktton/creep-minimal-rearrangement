#!/bin/bash
# Build mr_creep on a compute node (never on the frontend) and run the tests there.
# usage: cluster/build.sh [partition]   -> build/mr_creep_<partition>
P=${1:-knl_tacc}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
srun -p "$P" -N1 -n1 -c4 -J mr_build --time=00:20:00 bash -c "
	cd '$ROOT' && mkdir -p build &&
	g++ -O2 -std=c++17 -Wall -Wextra -fopenmp -march=native -isystem third_party -o build/mr_creep_$P src/mr_creep.cpp &&
	g++ -O2 -std=c++17 -fopenmp -march=native -isystem third_party -o build/check_$P tests/check.cpp &&
	build/check_$P"
