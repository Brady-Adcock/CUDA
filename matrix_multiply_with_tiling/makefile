
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


OBJS = wrappers.o matMultiply.o h_matMultiply.o d_matMultiply.o
.SUFFIXES: .cu .o .h 
.cu.o:
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

matMultiply: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -o matMultiply

matMultiply.o: matMultiply.cu h_matMultiply.h d_matMultiply.h config.h

h_matMultiply.o: h_matMultiply.cu h_matMultiply.h CHECK.h

d_matMultiply.o: d_matMultiply.cu d_matMultiply.h CHECK.h config.h

wrappers.o: wrappers.cu wrappers.h

clean:
	rm matMultiply *.o
