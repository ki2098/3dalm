windtunnel:
	mpic++ -acc -Minfo -Wall -fast src/windtunnel.cpp -o bin/windtunnel > make.log 2>&1

windtunnel_mpi:
	mpic++ -acc -Minfo -Wall -fast src/windtunnel_mpi.cpp -o bin/windtunnel_mpi > make.log 2>&1

2vtk:
	g++ -Wall -O2 src/2vtk.cpp -I/opt/vtk/9.3/include/vtk-9.3 -L/opt/vtk/9.3/lib -lvtkIOXML-9.3 -lvtkIOXMLParser-9.3 -lvtkCommonExecutionModel-9.3 -lvtkCommonDataModel-9.3 -lvtkCommonTransforms-9.3 -lvtkCommonMath-9.3 -lvtkCommonCore-9.3 -lvtksys-9.3 -lvtkkissfft-9.3 -lvtkCommonColor-9.3 -o bin/2vtk

reconstructor:
	g++ -Wall -O2 src/reconstructor.cpp -o bin/reconstructor