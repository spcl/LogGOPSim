import sys
import re
import glob
from goal import GoalComm


class AllprofParser:

    def __init__(self):
        self.tracenamePtrn = "pmpi-trace-rank-*.txt"
        self.comm = None
        self.verbose = True
        self.requests = []
        self.REQUEST_SIZE = 8

    def parseLine(self, rank, allprof_line):
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
                args = args.strip()
                args = args.split(":")
                args.append(rank)
                if self.verbose:
                    print("Parsing "+name+" with args "+str(args))
                getattr(self, name)(*args)
            else:
                raise NotImplementedError("Parsing of "+allprof_line.strip()+" is not implemented yet.")
        else:
            raise ValueError("The line "+allprof_line+" doesn't look like anything allprof should output!")

    def MPI_Initialized(self, tstart, flagptr, tend, rank):
        return # this doesn't modify the goal schedule
    
    def MPI_Init(self, tstart, argcptr, argvptr, tend, rank):
        return # this doesn't modify the goal schedule
    
    def MPI_Comm_size(self, tstart, comm, sizeptr, tend, rank):
        return # this doesn't modify the goal schedule
    
    def MPI_Comm_rank(self, tstart, comm, rankptr, tend, rank):
        return # this doesn't modify the goal schedule
    
    def MPI_Irecv(self, tstart, buf, count, datatype, src, tag, comm, req, tend, rank):
        ddtsize = self.getDDTSize(datatype)
        op = self.comm[rank].Recv(int(src), int(tag), int(count)*ddtsize)
        self.addRequest(rank, req, op)
        return #TODO handle splitted comms

    def MPI_Isend(self, tstart, buf, count, datatype, dst, tag, comm, req, tend, rank):
        ddtsize = self.getDDTSize(datatype)
        op = self.comm[rank].Send(int(dst), int(tag), int(count)*ddtsize)
        self.addRequest(rank, req, op)
        return #TODO handle splitted comms
    
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
    
    def MPI_Wait(self, tstart, requestptr, statusptr, tend, rank):
        calc = None
        op = self.findRequest(rank, int(requestptr))
        if op is None:
            print("Wait on a request we didn't see before - might be ok if the user initialized it to MPI_REQUEST_NULL, but also might mean request size is set to the wrong constant! -- check the code of the trace app!")
            return
        calc = self.comm[rank].Calc(0)
        calc.requires(op)

    def MPI_Barrier(self, tstart, comm, tend, rank):
        return #TODO implement

    def MPI_Wtime(self, tstart, tend, rank):
        return #this does not modify the goal schedule
    
    def MPI_Allreduce(tstart, sendbuf, recvbuf, count, datatype, op, comm, tend, rank):
        return #TODO implement
    
    def MPI_Reduce(*args):
        return #TODO implement
    
    def MPI_Finalize(self, tstart, tend, rank):
        return #this does not modify the goal schedule

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

    def TraceFileForRank(self, rank):
        fh = open(self.tracepath + "/" + "pmpi-trace-rank-" + str(rank) + ".txt")
        return fh


    def parseDir(self, tracepath, abortonerror=False):
      self.tracepath = tracepath
      files = glob.glob(tracepath + self.tracenamePtrn)
      self.comm = GoalComm(len(files))
      for rank in range(0, self.comm.CommSize()):
          self.requests.append({})
      for rank in range(0, self.comm.CommSize()):
          fh = self.TraceFileForRank(rank)
          while True:
            line = fh.readline()
            if not line:
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
                        print("There was a problem but we attempt to carry on: "+str(e))
          fh.close()
      return self.comm



if __name__ == "__main__":
    p = AllprofParser()
    comm = p.parseDir("./lulesh_8/", True)
    comm.write_goal()

