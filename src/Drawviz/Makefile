CXXFLAGS= -O0 -Wno-deprecated -Wall
CCFLAGS= -O0 
LDFLAGS= -lps -lboost_regex-mt

all:
	gengetopt < drawviz.ggo
	gcc $(CCFLAGS) -c *.c
	g++ $(CXXFLAGS) -c *.cpp
	g++ $(CXXFLAGS) *.o -o drawviz $(LDFLAGS)

clean:
	rm -f cmdline.o
	rm -f *.o
	rm -f drawviz
