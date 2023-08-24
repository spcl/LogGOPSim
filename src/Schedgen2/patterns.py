from math import ceil, log2
from goal import GoalComm


def binomialtree(comm_size, datasize, tag, dir="reduce"):
    comm = GoalComm(comm_size)
    for rank in range(0, comm_size):
        send = None
        recv = None
        for r in range(0, ceil(log2(comm_size))):
            peer = rank + pow(2, r)
            if (rank + pow(2, r) < comm_size) and (rank < pow(2, r)):
                if dir == "reduce":
                    recv = comm.Recv(size=datasize, src=peer, dst=rank, tag=tag)
                elif dir == "bcast":
                    send = comm.Send(size=datasize, dst=peer, src=rank, tag=tag)
                else:
                    raise ValueError(
                        "direction " + str(dir) + " in binomialtree not implemented."
                    )
            if (send is not None) and (recv is not None):
                send.requires(recv)
            peer = rank - pow(2, r)
            if (rank >= pow(2, r)) and (rank < pow(2, r + 1)):
                if dir == "reduce":
                    send = comm.Send(size=datasize, dst=peer, src=rank, tag=tag)
                if dir == "bcast":
                    recv = comm.Recv(size=datasize, src=peer, dst=rank, tag=tag)

    return comm


def dissemination(comm_size, datasize, tag):
    comm = GoalComm(comm_size)
    for rank in range(0, comm_size):
        dist = 1
        recv = None
        while dist < comm_size:
            send = comm.Send(
                src=rank,
                dst=(rank + dist + comm_size) % comm_size,
                size=datasize,
                tag=tag,
            )
            if recv is not None:
                send.requires(recv)
            recv = comm.Recv(
                src=(rank - dist + comm_size) % comm_size,
                dst=rank,
                size=datasize,
                tag=tag,
            )
            dist *= 2
    return comm


def ring_allreduce(comm_size, datasize, base_tag):
    comm = GoalComm(comm_size)
    for rank in range(0, comm_size):
        recv = None
        send = None
        chunk_size = (
            datasize // comm_size
            if datasize % comm_size == 0
            else datasize // comm_size + 1
        )
        for _ in range(2):
            # Phase 0: reduce-scatter, Phase 1: allgather
            for _ in range(0, comm_size - 1):
                send = comm.Send(
                    src=rank, dst=(rank + 1) % comm_size, size=chunk_size, tag=base_tag
                )
                if recv is not None:
                    send.requires(recv)
                recv = comm.Recv(
                    src=(rank - 1) % comm_size, dst=rank, size=chunk_size, tag=base_tag
                )
            # update tag for next phase
            base_tag += 1
    return comm
