CXX=g++
CXX_FLAGS=-O3 -Wall -march=native -std=c++11 -I../external/benchmark/include -L../external/benchmark/src -lbenchmark -lpthread

.PHONY: run_float run_double clean

run_float:
	$(CXX) run_float.cpp -o run_float $(CXX_FLAGS)

run_double:
	$(CXX) run_double.cpp -o run_double $(CXX_FLAGS)

clean:
	rm -rf run_float run_double
