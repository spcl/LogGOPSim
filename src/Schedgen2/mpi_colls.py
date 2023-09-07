from goal import GoalComm
from patterns import binomialtree, recdoub, ring, linear


def dissemination(comm_size, datasize, tag):
    #TODO: select or implement right pattern
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

def incast(comm_size: int, datasize: int, tag: int = 42, ptrn: str = "linear", unbalanced: bool = False, **kwargs):
    assert ptrn == "linear", "incast only supports the linear communication pattern"
    return linear(
        comm_size=comm_size,
        datasize=datasize,
        tag=tag,
        algorithm="incast",
        parallel=True,
        randomized_data=unbalanced,
        **kwargs
    )

def outcast(comm_size: int, datasize: int, tag: int = 42, ptrn: str = "linear", unbalanced: bool = False, **kwargs):
    assert ptrn == "linear", "outcast only supports the linear communication pattern"
    return linear(
        comm_size=comm_size,
        datasize=datasize,
        tag=tag,
        algorithm="outcast",
        parallel=True,
        randomized_data=unbalanced,
        **kwargs
    )

def reduce(comm_size: int, datasize: int, tag: int = 42, ptrn: str = "binomialtree", unbalanced: bool = False, **kwargs):
    if ptrn == "binomialtree":
        assert unbalanced == False, "binomialtree does not currently support randomized data"
        return binomialtree(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="reduce",
            **kwargs
        )
    elif ptrn == "linear":
        return linear(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="reduce",
            parallel=True,
            randomized_data=unbalanced,
            **kwargs
        )
    else:
        raise ValueError(f"reduce with pattern {ptrn} not implemented")

def bcast(comm_size: int, datasize: int, tag: int = 42, ptrn: str = "binomialtree", unbalanced: bool = False, **kwargs):
    if ptrn == "binomialtree":
        assert unbalanced == False, "binomialtree does not currently support randomized data"
        return binomialtree(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="bcast",
            **kwargs
        )
    elif ptrn == "linear":
        return linear(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="bcast",
            parallel=True,
            randomized_data=unbalanced,
            **kwargs
        )
    else:
        raise ValueError(f"bcast with pattern {ptrn} not implemented")

def allreduce(comm_size: int, datasize: int, tag: int = 42, ptrn: str = "recdoub", unbalanced: bool = False, **kwargs):
    assert unbalanced == False, "unbalanced data not currently supported"
    comms = [] # reduce-scatter and allgather
    if ptrn == "recdoub":
        comms.append(
            recdoub(
                comm_size=comm_size,
                datasize=datasize,
                tag=tag,
                algorithm="reduce-scatter",
                **kwargs
            )
        )
        comms.append(
            recdoub(
                comm_size=comm_size,
                datasize=datasize,
                tag=tag+comm_size,
                algorithm="allgather",
                **kwargs
            )
        )
    elif ptrn == "ring":
        for i in range(2):
            comms.append(
                ring(
                    comm_size=comm_size,
                    datasize=datasize,
                    tag=tag + (i * comm_size),
                    rounds=comm_size-1,
                    **kwargs
                )
            )
    else:
        raise ValueError(f"allreduce with pattern {ptrn} not implemented")
    comms[0].Append(comms[1])
    return comms[0]

def alltoall(comm_size: int, datasize: int, tag: int = 42, ptrn: str = "linear", unbalanced: bool = False, window_size: int = 0, **kwargs):
    if ptrn == "linear":
        return linear(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="alltoall",
            parallel=(window_size == 0),
            randomized_data=unbalanced,
            window_size=window_size,
            **kwargs
        )
    else:
        raise ValueError(f"alltoall with pattern {ptrn} not implemented")


def multi(collective: callable, num_comm_groups: int, comm_size: int, **kwargs):
    comm = GoalComm(comm_size * num_comm_groups)
    comms = comm.CommSplit(
        color=[i // comm_size for i in range(comm_size * num_comm_groups)],
        key=[i % comm_size for i in range(comm_size * num_comm_groups)],
    )
    for comm_split in comms:
        comm_collective = collective(comm_size=comm_size, **kwargs)
        comm_split.Append(comm_collective)
    return comm