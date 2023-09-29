import sys
import re
import glob
from goal import GoalComm

class AllprofParser:

    def __init__(self):
        self.tracenamePtrn = "pmpi-trace-rank-*.txt"
        self.comm = None
        self.parsable = []

    def parseLine(self, rank, allprof_line):
        # if the line is a comment, ignore it
        if re.match("#.*\n", allprof_line):
            return
        # if the line is whitespace ignore it
        if re.match("\s*\n", allprof_line):
            return
        # check if it matches any of the defined MPI functions

        raise(NotImplementedError("Parsing ["+allprof_line+"] is not implemented."))

    def MPI_Initialized(self, tstart, flagptr, tend):
        return

    def TraceFileForRank(self, rank):
        fh = open(self.tracepath + "/" + "pmpi-trace-rank-" + str(rank) + ".txt")
        return fh


    def parseDir(self, tracepath, abortonerror=False):
      self.tracepath = tracepath
      files = glob.glob(tracepath + self.tracenamePtrn)
      self.comm = GoalComm(len(files))
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

