INCLUDE_DIRS = -I/usr/include/opencv4
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g -fopenmp $(INCLUDE_DIRS) $(CDEFS)
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt

HFILES= 
CFILES= lanesnpeds.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.cpp=.o}

all: lanesnpeds

clean:
	-rm -f *.o *.d
	-rm -f lanesnpeds
	
lanesnpeds: lanesnpeds.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)



depend:

.cpp.o: $(SRCS)
	$(CC) $(CFLAGS) -c $<