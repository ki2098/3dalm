flags = -acc -Minfo=accel -Wall

plain_windtunnel:
	nvc++ $(flags) src/plain_windtunnel.cpp -o bin/plain_windtunnel