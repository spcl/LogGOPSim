The LogGOPSim Toolchain
=======================

The tools in this repository are centered around LogGOPSim, a network simulator
based on the LogGP model.

For a full explanation of this model, please see the referenced publications. But in
short this model (as implemented in LogGOPSim) provides the following:
  
  * Matching semantics similar to MPI, i.e., a send matches a specific receive, thus both sender and receiver can influence matching, and dependencies between recv and send operations can be expressed, thus real-world applications can be simulated (unlike other simulators which rely on predefined traffic patterns).
  * Nessages take a uniform amount of time between any pair of hosts, regardless of other traffic (there are extensions of LogGOPSim which change that), thus large-scale simulations can be performed relatively fast, compared to packet-based simulators.
  
Parts of the toolchain
======================

 * LogGOPSim: The simulator itself. It consumes a GOAL binary file, which specifies the actions (send and receive) of each host in the simulated network and produces a timing report, i.e., the time at which each host finishes its execution (among other data).
 * Schedgen:  While it is possible to write a GOAL file for LogGOPSim by hand, this is not advised. Instead, the Schedgen tool can be used to create such files. Schedgen can produce GOAL files for single MPI collective operations, but also allows to produce GOAL files which mimic the communication patterns observed in ML training workloads. It can also convert traces of MPI applications into the GOAL format. In case Schedgen does not offer the communication pattern you want to simulate, it can be extended using a C++ or Python API.
 * Schedgen2: An experimental re-implementation of Schedgen in Python - while this offers features that Schedgen lacks it misses many things stil.
 * Txt2bin: The output of Schedgen is produced in a human-readable text format, which makes it easy to debug schedules, however, for large scale simulations the limiting resource is memory/cache, thus we convert the GOAL file into a space-efficient binary format before feeding it into LogGOPSim. The txt2bin tool performs this conversion. When invoking LogGOPSim, the user has the option of allowing "destructive reading" of the binary schedule, i.e., the input file is memory mapped and modified during the execution to limit further reduce the amount of memory required during large simulations.
 * liballprof: A wrapper library around MPI which records all MPI calls, including their non-data arguments, the MPI traces produced can be converted into the GOAL format by Schedgen.


Building the toolchain
======================

On a recent Debian-based distro such as Ubuntu you can install the build dependencies with something like
```
sudo apt-get install cmake gengetopt re2c libgraphviz-dev python3 libclang-15-dev llvm-15-dev python3-clang-15 openmpi-bin openmpi-common libopenmpi-dev libunwind-dev
```
YMMV, but this is what we use in our CI pipeline.


This project uses cmake as its build tool:
```
 git clone [This repo]
 cd LogGOPSim
 mkdir build
 cd build
 cmake ../src/CMakeLists.txt
 make
```

Simple usage example
====================

```
  # we assume we are in the build folder, i.e., completed the steps above
 ./schedgen --commsize 20 --datasize 1024 --ptrn binomialtreereduce -o example.goal  # generate a GOAL text file for a simple pattern (a reduction using a binomial tree, for 20 hosts, each host contributing 1024 bytes)
 ./txt2bin -i example.goal -o example.bin   # convert the GOAL text file into the binary format required by LogGOPSim
 ./LogGOPSim -f example.bin                 # run LogGOPSim with default parameters (see output below, try running with --help to see how to change them)                                      
  LogGP network backend; size: 8 (1 CPUs, 1 NICs); L=2500, o=1500 g=1000, G=6, O=0, P=8, S=65535
  PERFORMANCE: Processes: 8 	 Events: 21 	 Time: 0 s 	 Speed: inf ev/s
  Times: 
  Host 0: 34914
  Host 1: 24776
  Host 2: 13138
  Host 3: 13138
  Host 4: 1500
  Host 5: 1500
  Host 6: 1500
  Host 7: 1500
```
