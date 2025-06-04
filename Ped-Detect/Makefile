INCLUDE_DIRS = -I/usr/include/opencv4
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g -fopenmp $(INCLUDE_DIRS) $(CDEFS)
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt

HFILES= 
CFILES= people_hog_framestop.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.cpp=.o}

all: people_hog_framestop

clean:
	-rm -f *.o *.d
	-rm -f people_hog_framestop
	
people_hog_framestop: people_hog_framestop.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)



depend:

.cpp.o: $(SRCS)
	$(CC) $(CFLAGS) -c $<
