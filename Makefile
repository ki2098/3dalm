cxx = g++
cxxversion = $(shell $(cxx) -dumpversion)

windtunnel:
	@echo gcc version = $(cxxversion)
ifeq ($(shell expr $(cxxversion) \< 8), 1)
	$(error needs gcc version 8 or above)
else ifeq ($(shell expr $(cxxversion) \= 8), 1)
	mpic++ src/windtunnel.cpp --std=c++17 -Wall -acc -Minfo -fast -lstdc++fs -o bin/windtunnel
else 
	mpic++ src/windtunnel.cpp --std=c++17 -Wall -acc -Minfo -fast -o bin/windtunnel
endif

2vtk:
	g++ -Wall -O2 src/2vtk.cpp -I/opt/vtk/9.3/include/vtk-9.3 -L/opt/vtk/9.3/lib -lvtkIOXML-9.3 -lvtkIOXMLParser-9.3 -lvtkCommonExecutionModel-9.3 -lvtkCommonDataModel-9.3 -lvtkCommonTransforms-9.3 -lvtkCommonMath-9.3 -lvtkCommonCore-9.3 -lvtksys-9.3 -lvtkkissfft-9.3 -lvtkCommonColor-9.3 -o bin/2vtk

reconstructor:
ifeq ($(shell expr $(cxxversion) \< 8), 1)
	$(error needs gcc version 8 or above)
else ifeq ($(shell expr $(cxxversion) \= 8), 1)
	g++ src/reconstructor.cpp --std=c++17 -Wall -O2 -lstdc++fs -o bin/reconstructor
else 
	g++ src/reconstructor.cpp -Wall -O2 -o bin/reconstructor
endif

all: windtunnel 2vtk reconstructor