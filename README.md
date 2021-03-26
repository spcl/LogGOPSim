# README file for the LogGOPSim simulator

Installing
----------

  * prerequisites to build: 
    - C++ compiler (e.g., g++)
    - re2c - http://re2c.org/
    - gengetopt - http://www.gnu.org/software/gengetopt/gengetopt.html
    - libagraph - http://www.graphviz.org/

  * build: 
    - optional: edit Makefile (change CXX and/or CXXFLAGS)
    - make

Running
-------

  * write or generate a GOAL schedule or use one of the example
    schedules (e.g., dissemination_16.goal or binary_tree_16.goal)

  * convert schedule to binary format using txt2bin:
    - e.g., txt2bin -i dissemination_16.goal -o dissemination_16.bin  

  * execute simulation with default parameters (see LogGOPSim --help for
    more options):
    - e.g., LogGOPSim -f dissemination_16.bin

  * interpret output:
    - for small simulations, each host end time is printed (22000ns for
      our example with default parameters)
    - for larger runs, only the maximum time is printed

Visualization
-------------

  * run LogGOPSim with -V <outfile> option:
    - e.g., LogGOPSim -f dissemination_16.bin -V viz.out

  * compile DrawViz (simple "make")
  
  * run DrawViz (only for smaller simulations):
    - e.g., drawviz -i viz.out -o viz.eps

  * view postscript output:
    - e.g., gv viz.eps

MPI Matching Data
------------------

  * run LogGOPSim with -qstat <outfile-prefix> option:
    - e.g., LogGOPSim -f dissemination_16.bin -stat mpi-matching will produce several
      files containing MPI match queue data with names that have the form mpi-matching.*.data

  * additional information on the MPI matching data is available in README-mpi-matching

Schedgen - automatic GOAL schedule generator
--------------------------------------------

  * compile SchedGen (simple "make")

  * run schedgen to generate schedules for collective operations:
    - e.g., schedgen -p binomialtreebcast -s 32 -o binary_tree_32.goal
      (generates a binomial tree brodacast pattern with 32 processes,
      the GOAL schedule can be converted to the binary simulator input
      with txt2bin)

  * run schedgen to generate schedules for application traces collected
    with liballprof-0.9:
    - traces need to be collected by linking liballprof as PMPI layer
      with an MPI application. Sample traces are included in the
      distribution in liballprof-samples
    - e.g., schedgen -p trace --traces liballprof-samples/sweep3d-2x2/pmpi-trace-rank-0.txt -o sweep-4.goal
    - convert and simulate:
      - e.g., txt2bin -i sweep-4.goal -o sweep-4.bin
              LogGOPSim -f sweep-4.bin
              
Liballprof-1.0 - Collecting traces from mpi applications
----------------------------------------------------
  (Using MPICH version 3.3.1 installed in /usr/local)

  * Prepare liballprof-1.0 
     'cd ~'
     'wget -qO- http://spcl.inf.ethz.ch/Research/Performance/LogGOPSim/liballprof-1.0.tgz | tar xzf -'
     'cd liballprof-1.0/'
     './configure'
     -> Add '-fPIC' to CFLAGS option in the Makefile
     
  * Compile liballprof-1.0
     'make'
     -> If conflic types error during make command, copy the mpi.h file to /usr/local/include/
     'make install'
          
  * Generate liballprof library
     'cd .libs/'
     'gcc -shared -o libConOs.so ../mpipclog.o ../sync.o -L~/liballprof-1.0/.libs -lclog'
     
  * Collect Traces using liballprof
    e.g., using a hello world mpi application
    - 'LD_PRELOAD=./libConOs.so mpirun -np 6 ../hello'
    The traces are generated in /tmp folder after running above command 
     
  

Citation
--------

Any published work which uses this software should include the following
citation:
----------------------------------------------------------------------
T. Hoefler, T. Schneider, A. Lumsdaine: LogGOPSim ­ Simulating
Large-Scale Applications in the LogGOPS Model
----------------------------------------------------------------------
