CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -fopenmp

build/mr_creep: src/mr_creep.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $<

test: build/mr_creep
	tests/run_tests.sh

clean:
	rm -rf build

.PHONY: test clean
