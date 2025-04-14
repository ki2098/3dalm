flags = -acc -Minfo=accel -O3

.bc:
	nvc++ $(flags) -c src/boundary_condition.cpp -o bin/boundary_condition.o

.cfd:
	nvc++ $(flags) -c src/cfd.cpp -o bin/cfd.o

.mv:
	nvc++ $(flags) -c src/mv.cpp -o bin/mv.o

.ls:
	nvc++ $(flags) -c src/pbicgstab.cpp -o bin/ls.o

plain_windtunnel: .bc .cfd .mv .ls
	nvc++ $(flags) -c src/plain_windtunnel.cpp -o bin/plain_windtunnel.o
	nvc++ -acc bin/boundary_condition.o bin/cfd.o bin/mv.o bin/ls.o bin/plain_windtunnel.o -o bin/plain_windtunnel