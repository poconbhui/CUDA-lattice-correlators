OUT=Tau.out

OBJ_DIR=obj/

CC=g++
NVCC=nvcc

CFLAGS=-c -Wall -O3 -std=c++0x
CUFLAGS=-c -O3 --compiler-options '-O3'
LINKFLAGS=-O3

LIBDIRS=-L/usr/local/cuda/lib64
LIBS=-lcuda -lcudart 

CUINCLUDES=-I/usr/local/cuda/include

SOURCES := $(wildcard *.cpp)
SOURCES += $(wildcard *.cu)

OBJS = $(SOURCES)
OBJS := $(patsubst %.cpp, %.o, $(OBJS))
OBJS := $(patsubst %.cu, %.o, $(OBJS))
OBJS := $(patsubst %.o, $(OBJ_DIR)%.o, $(OBJS))



all: $(OUT)

$(OUT) : $(OBJS)
	$(CC) $(D) $^ -o $@ $(LIBDIRS) $(LIBS) $(LINKFLAGS)

$(OBJ_DIR)%.o : %.cu
	$(NVCC) $(D) $(CUFLAGS) $< -o $@

$(OBJ_DIR)%.o : %.cpp
	$(CC) $(D) $(CFLAGS) $< -o $@ $(CUINCLUDES)

.PHONY : clean
clean :
	-rm $(OBJ_DIR)*
	
.PHONY : remake
remake : clean all
