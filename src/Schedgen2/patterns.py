from math import ceil, log2
from goal import GoalComm

def binomialtree(comm_size, datasize, tag, dir="reduce"):
    comm = GoalComm(comm_size)
    for rank in range(0, comm_size):
        send = None
        recv = None
        for r in range(0, ceil(log2(comm_size))):
            peer = rank + pow(2, r);
            if ((rank + pow(2, r) < comm_size) and (rank < pow(2, r))):
                if dir == "reduce":
                    recv = comm.Recv(size=datasize, src=peer, dst=rank, tag=tag);
                elif dir == "bcast":
                    send = comm.Send(size=datasize, dst=peer, src=rank, tag=tag);
                else:
                    raise ValueError("direction "+str(dir)+" in binomialtree not implemented.")
            if ((send is not None) and (recv is not None)):
                send.requires(recv);
            peer = rank - pow(2, r);
            if ((rank >= pow(2, r)) and (rank < pow(2, r + 1))):
                if dir == "reduce":
                    send = comm.Send(size=datasize, dst=peer, src=rank, tag=tag);
                if dir == "bcast":
                    recv = comm.Recv(size=datasize, src=peer, dst=rank, tag=tag);
                    
    return comm


def dissemination(comm_size, datasize, tag):
    comm = GoalComm(comm_size)
    for rank in range(0, comm_size):
        dist = 1
        while (dist < comm_size):
            send = comm.Send(src=rank, dst=(rank+dist+comm_size)%comm_size, size=datasize, tag=tag)
            if revc is not None:
                send.requires(recv)
            recv - comm.Recv(src=(rank-dist+comm_size)%comm_size, dst=rank, size=datasize, tag=tag)
            dist *= 2
    return comm	
