import sys
import argparse
from goal import GoalComm
from patterns import binomialtree, dissemination

parser = argparse.ArgumentParser(description="Generate GOAL Schedules.")
parser.add_argument("--ptrn", dest="ptrn", choices=["binomialtreereduce", "binarytreebcast"])
parser.add_argument("--commsize", dest="commsize", type=int, default=8, help="Size of the communicator")
parser.add_argument("--datasize", dest="datasize", type=int, default=8, help="Size of the data, i.e., for reduce operations")

args = parser.parse_args()



if args.ptrn == "binomialtreereduce":
    g = binomialtree(args.commsize, args.datasize, 42, "reduce")
    g.write_goal(sys.stdout)

if args.ptrn == "binomialtreebcast":
    g = binomialtree(args.commsize, args.datasize, 42, "bcast")
    g.write_goal(sys.stdout)

if args.ptrn == "dissemination":
    g = dissemination(args.commsize, args.datasize, 42)
    g.write_goal(sys.stdout)
