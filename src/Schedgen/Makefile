CXXFLAGS=-g -O3 -Wno-deprecated -Wall -std=c++11
CCFLAGS=-g -O3 -g
LDFLAGS=-g -O3 -g -lboost_iostreams -L/opt/homebrew/lib/ 

force: all

schedgen_cmdline.c: schedgen_cmdline.ggo
	gengetopt -i $< -F schedgen_cmdline

%.o: %.cpp *hpp *h
	${CXX} $(CXXFLAGS) -c $<

%.o: %.c *h
	${CC} $(CCFLAGS) -c $<

all: buffer_element.o  schedgen_cmdline.o  process_trace.o  schedgen_cmdline.ggo  schedgen.o 
	${CXX} $(CXXFLAGS) *.o -o schedgen $(LDFLAGS)

clean:
	rm -f *.o
	rm -f schedgen_cmdline.c schedgen_cmdline.h
	rm -f schedgen
