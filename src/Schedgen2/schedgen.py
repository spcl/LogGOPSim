import sys
import argparse
from math import ceil, log2
from goal import GoalComm

parser = argparse.ArgumentParser(description="Generate GOAL Schedules.")
parser.add_argument("--ptrn", dest="ptrn", choices=["binomialtreereduce", "binarytreebcast"])
parser.add_argument("--commsize", dest="commsize", type=int, default=8, help="Size of the communicator")
parser.add_argument("--datasize", dest="datasize", type=int, default=8, help="Size of the data, i.e., for reduce operations")

args = parser.parse_args()

def generate_binomialtreereduce(comm_size, datasize, tag):
    comm = GoalComm(comm_size)
    for rank in range(0, comm_size):
        send = None
        recv = None
        for r in range(0, ceil(log2(comm_size))):
            peer = rank + pow(2, r);
            if ((rank + pow(2, r) < comm_size) and (rank < pow(2, r))):
                recv = comm.Recv(size=datasize, src=peer, dst=rank, tag=tag);
            if ((send is not None) and (recv is not None)):
                send.requires(recv);
            peer = rank - pow(2, r);
            if ((rank >= pow(2, r)) and (rank < pow(2, r + 1))):
                send = comm.Send(size=datasize, dst=peer, src=rank, tag=tag);
    return comm


if args.ptrn == "binomialtreereduce":
    g = generate_binomialtreereduce(args.commsize, args.datasize, 42)
    g.write_goal(sys.stdout)
