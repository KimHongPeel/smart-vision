CX = g++
CC = gcc

CXFLAGS = -g -Wall
CCFLAGS = -g -Wall

DXLFLAGS = -I/usr/local/include/dynamixel_sdk
DXLFLAGS += -ldxl_x86_c
DXLFLAGS += -lrt

CVFLAGS = `pkg-config opencv4 --cflags --libs`

BUILDFLAGS = $(CVFLAGS)
BUILDFLAGS += $(DXLFLAGS)

TARGET = lane_follow
OBJS = main.o dxl.o

$(TARGET) : $(OBJS)
	$(CX) $(CXFLAGS) -o $(TARGET) $(OBJS) $(BUILDFLAGS)
main.o : main.cpp
	$(CX) $(CXFLAGS) -c main.cpp $(BUILDFLAGS)
dxl.o : dxl.h dxl.c
	$(CC) $(CCFLAGS) -c dxl.c $(BUILDFLAGS)

.PHONY: all clean
 
all: $(TARGET)
 
clean:
	rm -rf $(TARGET) $(OBJS)
