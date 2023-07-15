CXXFLAGS=-g -O3 -Wno-deprecated -Wall -std=c++11
CCFLAGS=-g -O3 -g
LDFLAGS=-g -O3 -g -lboost_iostreams -L/opt/homebrew/lib/ 

force: all

cmdline.c: schedgen.ggo
	gengetopt < $<

%.o: %.cpp *hpp *h
	${CXX} $(CXXFLAGS) -c $<

%.o: %.c *h
	${CC} $(CCFLAGS) -c $<

all: buffer_element.o  cmdline.o  process_trace.o  schedgen.ggo  schedgen.o 
	${CXX} $(CXXFLAGS) *.o -o schedgen $(LDFLAGS)

clean:
	rm -f *.o
	rm -f cmdline.c cmdline.h
	rm -f schedgen
