flags = -acc -Minfo=accel -O3

plain_windtunnel:
	nvc++ $(flags) src/plain_windtunnel.cpp -o bin/plain_windtunnel