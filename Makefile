windtunnel:
	nvc++ -acc -Minfo=accel -Wall src/windtunnel.cpp -o bin/windtunnel > make.log 2>&1

windtunnel2:
	nvc++ -acc -Minfo=accel -Wall src/windtunnel2.cpp -o bin/windtunnel2 > make.log 2>&1

windtunnel_mpi:
	mpic++ -acc -Minfo -Wall src/windtunnel_mpi.cpp -o bin/windtunnel_mpi > make.log 2>&1