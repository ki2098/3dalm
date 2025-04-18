flags = -Minfo=accel -Wall

plain_windtunnel:
	nvc++ $(flags) -acc src/plain_windtunnel.cpp -o bin/plain_windtunnel > make.log 2>&1
