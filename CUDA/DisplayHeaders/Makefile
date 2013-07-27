OBJ_DIR = obj/

CC = g++
NVCC = nvcc

CFLAGS=-c -Wall
CUFLAGS=-c

LIBDIRS=-L/usr/local/cuda/lib64
LIBS=-lcuda -lcudart -lboost_thread-mt

SOURCES := $(wildcard *.cpp)
SOURCES += $(wildcard *.cu)

OBJS = $(SOURCES)
OBJS := $(patsubst %.cpp, %.o, $(OBJS))
OBJS := $(patsubst %.cu, %.o, $(OBJS))
OBJS := $(patsubst %.o, $(OBJ_DIR)%.o, $(OBJS))



all: Headers.out

Headers.out : $(OBJS)
	$(CC) $^ -o $@ $(LIBDIRS) $(LIBS)

$(OBJ_DIR)%.o : %.cu
	$(NVCC) $(CUFLAGS) $< -o $@

$(OBJ_DIR)%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@



	
