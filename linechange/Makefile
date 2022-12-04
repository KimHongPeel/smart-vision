CC = gcc
CCFLAGS = -g -Wall

DXLFLAGS = -I/usr/local/include/dynamixel_sdk
DXLFLAGS += -ldxl_x86_c
DXLFLAGS += -lrt

TARGET = dxltest
OBJS = main.o dxl.o
$(TARGET) :  $(OBJS)
	$(CC) $(CCFLAGS) -o $(TARGET) $(OBJS) $(DXLFLAGS)
main.o : main.c
	$(CC) $(CCFLAGS) -c main.c $(DXLFLAGS)
dxl.o : dxl.h dxl.c
	$(CC) $(CCFLAGS) -c dxl.c $(DXLFLAGS)

.PHONY: all clean

all: $(TARGET)

clean:
	rm -rf $(TARGET) $(OBJS)


