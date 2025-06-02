cxx = g++
cxxversion = $(shell $(cxx) -dumpversion)
vtk_home = /opt/vtk/9.3

.bin_dir:
	mkdir -p bin

solver: .bin_dir
	@echo gcc version = $(cxxversion)
ifeq ($(shell expr $(cxxversion) \< 8), 1)
	$(error needs gcc version 8 or above)
else ifeq ($(shell expr $(cxxversion) \= 8), 1)
	mpic++ src/solver.cpp --std=c++17 -Wall -acc -Minfo -fast -lstdc++fs -o bin/solver
else 
	mpic++ src/solver.cpp --std=c++17 -Wall -acc -Minfo -fast -o bin/solver
endif

2vtk: .bin_dir
	g++ -O2 src/2vtk.cpp -I$(vtk_home)/include/vtk-9.3 -L$(vtk_home)/lib -lvtkIOXML-9.3 -lvtkIOXMLParser-9.3 -lvtkCommonExecutionModel-9.3 -lvtkCommonDataModel-9.3 -lvtkCommonTransforms-9.3 -lvtkCommonMath-9.3 -lvtkCommonCore-9.3 -lvtksys-9.3 -lvtkkissfft-9.3 -lvtkCommonColor-9.3 -o bin/2vtk

reconstructor: .bin_dir
ifeq ($(shell expr $(cxxversion) \< 8), 1)
	$(error needs gcc version 8 or above)
else ifeq ($(shell expr $(cxxversion) \= 8), 1)
	g++ src/reconstructor.cpp --std=c++17 -O2 -lstdc++fs -o bin/reconstructor
else 
	g++ src/reconstructor.cpp -O2 -o bin/reconstructor
endif

all: solver 2vtk reconstructor