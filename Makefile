.bc:
	nvc++ -acc -Minfo -O3 -c src/boundary_condition.cpp -o bin/boundary_condition.o

.cfd:
	nvc++ -acc -Minfo -O3 -c src/cfd.cpp -o bin/cfd.o

.mv:
	nvc++ -acc -Minfo -O3 -c src/mv.cpp -o bin/mv.o

.ls:
	nvc++ -acc -Minfo -O3 -c src/pbicgstab.cpp -o bin/ls.o

plain_windtunnel: .bc .cfd .mv .ls
	nvc++ -acc -Minfo -O3 -c src/plain_windtunnel.cpp -o bin/plain_windtunnel.o
	nvc++ -acc -Minfo -O3 bin/boundary_condition.o bin/cfd.o bin/mv.o bin/ls.o bin/plain_windtunnel.o -o bin/plain_windtunnel