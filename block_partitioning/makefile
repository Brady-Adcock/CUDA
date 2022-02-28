NVCC = /usr/bin/nvcc
CC = g++

#No optmization flags
#--compiler-options sends option to host compiler; -Wall is all warnings
#NVCCFLAGS = -c --compiler-options -Wall

#Optimization flags: -O2 gets sent to host compiler; -Xptxas -O2 is for
#optimizing PTX
NVCCFLAGS = -c -O2 -Xptxas -O2 --compiler-options -Wall

#Flags for debugging
#NVCCFLAGS = -c -G --compiler-options -Wall --compiler-options -g

OBJS = wrappers.o vecScalarMult.o h_vecScalarMult.o d_vecScalarMult.o
.SUFFIXES: .cu .o .h 
.cu.o:
	$(NVCC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

vecScalarMult: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -o vecScalarMult

vecScalarMult.o: vecScalarMult.cu h_vecScalarMult.h d_vecScalarMult.h config.h

h_vecScalarMult.o: h_vecScalarMult.cu h_vecScalarMult.h CHECK.h

d_vecScalarMult.o: d_vecScalarMult.cu d_vecScalarMult.h CHECK.h config.h

wrappers.o: wrappers.cu wrappers.h

clean:
	rm vecScalarMult *.o
