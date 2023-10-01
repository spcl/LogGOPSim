import sys
import ast
import re
import glob
import argparse
import os
from mpi_colls import allreduce, alltoall
from goal import GoalComm


class AllprofParser:

    def __init__(self, requestsize=8, verbose=False):
        self.comm = None
        self.verbose = verbose
        self.requests = []    # for each comm_world rank, this holds one dict, mapping reqptrs to ops
        self.REQUEST_SIZE = requestsize # this is the size in bytes of a MPI_Request, i.e., in waitall we step through the array using this stepsize
        self.last_op = {}     # one pair (op, endtime) per comm_world rank
    
    def getLastOp(self, rank: int):
        if rank not in self.last_op:
            return None
        return self.last_op[rank]
    
    def setLastOp(self, rank: int, op, endtime: int):
        self.last_op[rank] = (op, endtime)

    def parseLine(self, rank: int, allprof_line: str):
        # if the line is a comment, ignore it
        if re.match("#.*\n", allprof_line):
            return
        # if the line is whitespace ignore it
        if re.match("\s*\n", allprof_line):
            return
        # check if it matches any of the defined MPI functions
        m = re.match("(MPI_.+?):(.+:(\d+|-))\n", allprof_line)
        if m:
            name = m.group(1)
            args = m.group(2)
            if hasattr(self, name):
                args = args.strip().split(":")
                # turn args into ints where possible (ddts, comms, ... are not ints!)
                newargs = []
                for arg in args:
                    newarg = 0
                    try:
                        newarg = int(arg)
                    except:
                        newarg = arg
                    newargs.append(newarg)
                args = newargs
                args.append(rank)
                if self.verbose:
                    print("Parsing "+name+" with args "+str(args))
                # for each line we get its start and end time (first and last elem in args)
                # we add a calc of the size of the difference between the endtime of the last
                # operation on rank and the starttime to account for any computation that might
                # have happened between calls - we init last_op in MPI_Init, so it might be None
                if self.getLastOp(rank) is not None:
                    tstart = int(args[0])
                    tend = int(args[-1])
                    last_op, last_endtime = self.getLastOp(rank)
                    newCalc = self.comm[rank].Calc(tstart - last_endtime)
                    newCalc.requires(last_op)
                    self.setLastOp(rank, newCalc, tend)
                newcomm = getattr(self, name)(*args)
                if newcomm is not None:
                    # append rank of newcomm to self.comm, however all independent ops in newcomm depend on last_op
                    # and the new last_op becomes the last op in newcomm (if there is only one, otherwise we make a calc of size 0)
                    self.comm[rank].Append(newcomm[rank], dependOn=self.getLastOp(rank)[0])
                    lastop = None
                    l = newcomm[rank].LastOps()
                    if len(l) == 1:
                        lastop = l[0]
                    else:
                        lastop = self.comm[rank].Calc(0)
                        lastop.requires(self.getLastOp(rank)[0]) # just to be save in case newcomm is empty
                        for o in l:
                            lastop.requires(o)
                    self.setLastOp(rank, lastop, args[-1])
            else:
                raise NotImplementedError("Parsing of "+allprof_line.strip()+" is not implemented yet.")
        else:
            raise ValueError("The line "+allprof_line+" doesn't look like anything allprof should output!")

    def MPI_Initialized(self, tstart, flagptr, tend, rank):
        return None # this doesn't modify the goal schedule
    
    def MPI_Init(self, tstart, argcptr, argvptr, tend, rank: int):
        self.setLastOp(rank, self.comm[rank].Calc(tend), tend)
        return None
    
    def MPI_Comm_size(self, tstart, comm, sizeptr, tend, rank):
        return None
    
    def MPI_Comm_rank(self, tstart, comm, rankptr, tend, rank):
        return None
    
    def MPI_Irecv(self, tstart, buf, count, datatype, src, tag, comm, req, tend, rank):
        g = GoalComm(self.comm.CommSize())
        ddtsize = self.getDDTSize(datatype)
        op = g[rank].Recv(int(src), int(tag), int(count)*ddtsize)
        self.addRequest(rank, req, op)
        return g #TODO handle splitted comms

    def MPI_Isend(self, tstart, buf, count, datatype, dst, tag, comm, req, tend, rank):
        g = GoalComm(self.comm.CommSize())
        ddtsize = self.getDDTSize(datatype)
        op = g[rank].Send(int(dst), int(tag), int(count)*ddtsize)
        self.addRequest(rank, req, op)
        return g #TODO handle splitted comms
    
    def MPI_Waitall(self, tstart, count, requestptr, statusptr, tend, rank):
        calc = None
        for ridx in range(0, int(count)):
            request = int(requestptr)+ridx*self.REQUEST_SIZE
            op = self.findRequest(rank, request)
            if op is None:
                print("Waitall on a request we didn't see before - might be ok if the user initialized it to MPI_REQUEST_NULL, but also might mean request size is set to the wrong constant! -- check the code of the trace app!")
                continue
            if calc is None:
                calc = self.comm[rank].Calc(0)
            calc.requires(op)
        # Waitall directly modifies self.comm, thus returns None and we need to handle deps from/on last op in here
        calc.requires(self.getLastOp(rank)[0])
        self.setLastOp(rank, calc, tend)
        return None
    
    def MPI_Wait(self, tstart, requestptr, statusptr, tend, rank):
        calc = None
        op = self.findRequest(rank, requestptr)
        if op is None:
            print("Wait on a request we didn't see before - might be ok if the user initialized it to MPI_REQUEST_NULL, but also might mean request size is set to the wrong constant! -- check the code of the trace app!")
            return
        calc = self.comm[rank].Calc(0)
        calc.requires(op)
         # Wait directly modifies self.comm, thus returns None and we need to handle deps from/on last op in here
        calc.requires(self.getLastOp(rank)[0])
        self.setLastOp(rank, calc, tend)
        return None

    def MPI_Barrier(self, tstart, comm, tend, rank):
        return alltoall(datasize=0, comm_size=self.comm.CommSize())

    def MPI_Wtime(self, tstart, tend, rank):
        return None #this does not modify the goal schedule
    
    def MPI_Allreduce(self, tstart, sendbuf, recvbuf, count, datatype, op, comm, tend, rank):
        datasize = self.getDDTSize(datatype) * count
        return allreduce(datasize, self.comm.CommSize())
    
    def MPI_Finalize(self, tstart, tend, rank):
        return None #this does not modify the goal schedule

    def addRequest(self, rank, req, op):
        self.requests[rank][int(req)] = op

    def findRequest(self, rank, req):
        if int(req) in self.requests[rank]:
            op = self.requests[rank][int(req)]
            return op
        return None
    
    def deleteRequest(self, rank, req):
        if int(req) in self.requests[rank]:
            self.requests[rank].pop(int(req))

    def getDDTSize(self, ddtstr):
        return int(ddtstr.split(",")[1])

    def parseDir(self, tracepath, nameptrn="pmpi-trace-rank-*.txt", abortonerror=False):
      self.tracepath = tracepath
      searchpath = os.path.join(tracepath, nameptrn)
      files = glob.glob(searchpath)
      if len(files) < 1:
          raise ValueError("No tracefiles found at path "+str(searchpath))
      self.comm = GoalComm(len(files))
      for rank in range(0, self.comm.CommSize()):
          self.requests.append({})
      for rank in range(0, self.comm.CommSize()):
          file_name = str(rank).join(nameptrn.split("*"))
          fh = open(os.path.join(tracepath, file_name), "r")
          while True:
            line = fh.readline()
            if not line:
                if self.verbose:
                    print("Finished parsing ranks "+str(rank)+" trace.")
                break
            else:
                try:
                    self.parseLine(rank, line)
                except Exception as e:
                    if abortonerror:
                        raise e
                        sys.exit(1)
                    else:
                        if self.verbose:
                            print("There was a problem but we attempt to carry on: "+str(e))
          fh.close()
      return self.comm



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Schedgen2 Trace Parser',
                    description='Reads an MPI trace in liballprof format and outputs a GOAL schedule (or a graphical representation of it).')
    parser.add_argument('-v', '--verbose', action='store_true', help="Be more verbose, i.e., print progress info.")
    parser.add_argument('-i', '--tracedir', required=True, help="Path to the directory containing the individual traces, each tracefile name follows nameptrn.")
    parser.add_argument('-n', '--nameptrn', default="pmpi-trace-rank-*.txt", help="Filename of traces, use * to indicate rank id (in MPI_COMM_WORLD), defaults to pmpi-trace-rank*.txt")
    parser.add_argument('-f', '--output-format', default="goal", choices=["goal", "graphviz"], help="Output format, either goal or graphviz, defaults to goal")
    parser.add_argument('-o', '--outfile', default="-", help="Output file name, use - for stdout (if verbose mode is on progress will be printed to stdout), defaults to -.")
    parser.add_argument('-r', '--requestsize', default=8, help="Size of an MPI_REQUEST in bytes, defaults to 8.")
    parser.add_argument('-a', '--abortonerror', action='store_true', help="By default we ignore errors such as not implemented MPI functions. Use this flag to abort on such errors.")
    args = parser.parse_args()
    p = AllprofParser(requestsize=args.requestsize, verbose=args.verbose)
    comm = p.parseDir(args.tracedir, nameptrn=args.nameptrn, abortonerror=args.abortonerror)
    outfile = sys.stdout
    if args.outfile != "-":
        outfile = open(args.outfile, "w")
    comm.write_goal(fh=outfile, format=args.output_format)
    outfile.close()

