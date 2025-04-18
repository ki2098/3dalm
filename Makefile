flags = -Minfo=accel -Wall

plain_windtunnel_gpu:
	nvc++ $(flags) -acc src/plain_windtunnel.cpp -o bin/plain_windtunnel_gpu

plain_windtunnel_cpu:
	nvc++ $(flags) src/plain_windtunnel.cpp -o bin/plain_windtunnel_cpu