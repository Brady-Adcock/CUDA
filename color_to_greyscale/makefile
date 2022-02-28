NVCC = /usr/bin/nvcc
CC = g++

#No optmization flags
#--compiler-options sends option to host compiler; -Wall is all warnings
#NVCCFLAGS = -c --compiler-options -Wall

#Optimization flags: -O2 gets sent to host compiler; -Xptxas -O2 is for
#optimizing PTX
#NVCCFLAGS = -c -O2 -Xptxas -O2 --compiler-options -Wall

#Flags for debugging
NVCCFLAGS = -c -G --compiler-options -Wall --compiler-options -g

OBJS = greyscaler.o wrappers.o h_colorToGreyscale.o d_colorToGreyscale.o
.SUFFIXES: .cu .o .h 
.cu.o:
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

all: greyscaler generate

greyscaler: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -ljpeg -o greyscaler

greyscaler.o: greyscaler.cu wrappers.h h_colorToGreyscale.h d_colorToGreyscale.h

h_colorToGreyscale.o: h_colorToGreyscale.cu h_colorToGreyscale.h CHECK.h

d_colorToGreyscale.o: d_colorToGreyscale.cu d_colorToGreyscale.h CHECK.h

wrappers.o: wrappers.cu wrappers.h

generate: generate.c
	gcc -O2 generate.c -o generate -ljpeg

clean:
	rm generate greyscaler *.o *.jpg
