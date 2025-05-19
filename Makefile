cxx = g++
cxxversion = $(shell $(cxx) -dumpversion)

windtunnel:
	@echo gcc version = $(cxxversion)
ifeq ($(shell expr $(cxxversion) \< 8), 1)
	$(error needs gcc version 8 or above)
else ifeq ($(shell expr $(cxxversion) \= 8), 1)
	mpic++ -std=c++17 -Wall -acc -Minfo -fast src/windtunnel.cpp -lstdc++fs -o bin/windtunnel > make.log 2>&1
endif
	mpic++ -std=c++17 -Wall -acc -Minfo -fast src/windtunnel.cpp -o bin/windtunnel > make.log 2>&1

2vtk:
	g++ -Wall -O2 src/2vtk.cpp -I/opt/vtk/9.3/include/vtk-9.3 -L/opt/vtk/9.3/lib -lvtkIOXML-9.3 -lvtkIOXMLParser-9.3 -lvtkCommonExecutionModel-9.3 -lvtkCommonDataModel-9.3 -lvtkCommonTransforms-9.3 -lvtkCommonMath-9.3 -lvtkCommonCore-9.3 -lvtksys-9.3 -lvtkkissfft-9.3 -lvtkCommonColor-9.3 -o bin/2vtk

reconstructor:
	g++ -Wall -O2 src/reconstructor.cpp -o bin/reconstructor