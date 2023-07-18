CXX=g++
CXXFLAGS= -std=c++11 -O0 -g -pedantic -Wno-deprecated -Wall -Wno-long-long -I/opt/homebrew/include/
CCFLAGS= -O0 -g 
LDFLAGS= -L/opt/homebrew/lib/ -lcgraph -g

AUTOGEN_SRC= loggopsim_cmdline.c loggopsim_cmdline.h txt2bin_cmdline.h txt2bin_cmdline.c  
LOGGOPSIM_OBJECTS= LogGOPSim.o 
HLPR_OBJECTS= loggopsim_cmdline.o 
ALL_OBJECTS= $(LOGGOPSIM_OBJECTS) $(HLPR_OBJECTS)
BINARY= LogGOPSim

all: $(ALL_OBJECTS) $(AUTOGEN_SRC) txt2bin
	$(CXX) $(CXXFLAGS) $(ALL_OBJECTS) -o $(BINARY) $(LDFLAGS)

txt2bin:
	re2c -o txt2bin.cpp txt2bin.re
	gengetopt -F txt2bin_cmdline -i txt2bin_cmdline.ggo
	$(CXX) -g -O3 txt2bin.cpp txt2bin_cmdline.c -o txt2bin

loggopsim_cmdline.c: loggopsim_cmdline.ggo 
	gengetopt -F loggopsim_cmdline -i loggopsim_cmdline.ggo

loggopsim_cmdline.h: loggopsim_cmdline.ggo
	gengetopt -F loggopsim_cmdline -i loggopsim_cmdline.ggo

txt2bin_cmdline.c: txt2bin_cmdline.ggo 
	gengetopt -F txt2bin_cmdline -i txt2bin_cmdline.ggo

txt2bin_cmdline.h: txt2bin_cmdline.ggo
	gengetopt -F txt2bin_cmdline -i txt2bin_cmdline.ggo



%.o: %.cpp $(AUTOGEN_SRC) *.hpp 
	$(CXX) $(CXXFLAGS) -c $<

%.o: %.c $(AUTOGEN_SRC) *.h
	$(CXX) $(CCFLAGS) -c $<

clean:
	rm -f $(AUTOGEN_SRC)
	rm -f $(ALL_OBJECTS)
	rm -f $(BINARY)
	rm -f txt2bin.cpp txt2bin bin2txt bin2dot simtest
	rm -f cmdline_txt2bin.*
