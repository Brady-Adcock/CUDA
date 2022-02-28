NVCC = /usr/bin/nvcc
CC = g++

#No optmization flags
#--compiler-options sends option to host compiler; -Wall is all warnings
NVCCFLAGS = -c --compiler-options -Wall

#Optimization flags: -O2 gets sent to host compiler; -Xptxas -O2 is for
#optimizing PTX 
#NVCCFLAGS = -c -O2 -Xptxas -O2 --compiler-options -Wall 

#Flags for debugging
#NVCCFLAGS = -c -G --compiler-options -Wall --compiler-options -g

OBJS = wrappers.o vecAdd.o h_vecAdd.o d_vecAdd.o
.SUFFIXES: .cu .o .h 
.cu.o:
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

vecAdd: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -o vecAdd

vecAdd.o: vecAdd.cu

h_vecAdd.o: h_vecAdd.cu h_vecAdd.h CHECK.h

d_vecAdd.o: d_vecAdd.cu d_vecAdd.h CHECK.h

wrappers.o: wrappers.cu wrappers.h

clean:
	rm vecAdd *.o
