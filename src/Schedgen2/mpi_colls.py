from math import log2, ceil
import random

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
    if ptrn == "recdoub":
        return recdoub(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="allreduce",
            **kwargs
        )
    elif ptrn == "ring":
        return ring(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="allreduce",
            **kwargs
        )
    else:
        raise ValueError(f"allreduce with pattern {ptrn} not implemented")

def recdoub_allreduce(comm, comm_size, datasize, tag, ctd=0):
    num_steps = int(log2(comm_size))
    for rank in range(0, comm_size):
        # Reduce-scatter
        sources = [rank ^ (2**i) for i in range(num_steps)]
        destinations = sources
        data_sizes_receive = [datasize // (2**i) for i in range(1, num_steps + 1)]
        data_sizes_send = data_sizes_receive
        tags = [tag + i for i in range(num_steps)]
        dependency = iterative_send_recv(
            comm,
            rank,
            sources,
            destinations,
            data_sizes_receive,
            data_sizes_send,
            tags,
            compute_time_dependency=ctd,
        )

        # Allgather
        sources = sources[::-1]
        destinations = sources
        data_sizes_receive = data_sizes_receive[::-1]
        data_sizes_send = data_sizes_send[::-1]
        tags = [tag + num_steps + i for i in range(num_steps)]
        iterative_send_recv(
            comm,
            rank,
            sources,
            destinations,
            data_sizes_receive,
            data_sizes_send,
            tags,
            last_dependency=dependency,
            compute_time_dependency=ctd,
        )


def ring_allreduce(comm, comm_size, datasize, tag, ctd=0):
    for rank in range(0, comm_size):
        chunk_size = (
            datasize // comm_size
            if datasize % comm_size == 0
            else datasize // comm_size + 1
        )
        sources = [(rank - 1) % comm_size] * (comm_size - 1)
        destinations = [(rank + 1) % comm_size] * (comm_size - 1)
        data_sizes_receive = [chunk_size] * (comm_size - 1)
        data_sizes_send = [chunk_size] * (comm_size - 1)
        tags = [tag + i for i in range(comm_size - 1)]
        dependency = iterative_send_recv(
            comm,
            rank,
            sources,
            destinations,
            data_sizes_receive,
            data_sizes_send,
            tags,
            compute_time_dependency=ctd,
        )
        tags = [tag + comm_size - 1 + i for i in range(comm_size - 1)]
        iterative_send_recv(
            comm,
            rank,
            destinations,
            sources,
            data_sizes_send,
            data_sizes_receive,
            tags,
            last_dependency=dependency,
            compute_time_dependency=ctd,
        )


def allreduce(algorithm, comm_size, datasize, tag, ctd=0, **kwargs):
    comm = GoalComm(comm_size)
    if algorithm == "ring":
        ring_allreduce(comm, comm_size, datasize, tag, ctd)
    elif algorithm == "recdoub":
        recdoub_allreduce(comm, comm_size, datasize, tag, ctd)
    elif algorithm == "datasize_based":
        if datasize < 4096:
            recdoub_allreduce(comm, comm_size, datasize, tag, ctd)
        else:
            ring_allreduce(comm, comm_size, datasize, tag, ctd)
    else:
        raise ValueError(f"allreduce algorithm {algorithm} not implemented")
    return comm


def multi_allreduce(algorithm, num_comm_groups, comm_size, **kwargs):
    comm = GoalComm(comm_size * num_comm_groups)
    comms = comm.CommSplit(
        color=[i // comm_size for i in range(comm_size * num_comm_groups)],
        key=[i % comm_size for i in range(comm_size * num_comm_groups)],
    )
    for comm_split in comms:
        allreduce(algorithm, comm_split.CommSize(), **kwargs)
    return comm


def windowed_alltoall(comm, window_size, comm_size, datasize, tag, **kwargs):
    for rank in range(0, comm_size):
        sources = [(rank - step) % comm_size for step in range(1, comm_size)]
        destination = [(rank + step) % comm_size for step in range(1, comm_size)]
        data_sizes_receive = [datasize] * (comm_size - 1)
        data_sizes_send = [datasize] * (comm_size - 1)

        windowed_send_recv(
            comm,
            rank,
            sources,
            destination,
            data_sizes_receive,
            data_sizes_send,
            window_size,
            tag,
        )


def balanced_alltoall(comm, comm_size, datasize, tag, **kwargs):
    for rank in range(0, comm_size):
        sources = [(rank - step) % comm_size for step in range(1, comm_size)]
        destination = [(rank + step) % comm_size for step in range(1, comm_size)]
        data_sizes_receive = [datasize] * (comm_size - 1)
        data_sizes_send = [datasize] * (comm_size - 1)

        parallel_send_recv(
            comm, rank, sources, destination, data_sizes_receive, data_sizes_send, tag
        )


def unbalanced_alltoall(comm, comm_size, datasize, tag, **kwargs):
    datasizes_randomized = [
        [
            datasize + int(0.1 * random.randint(-datasize, datasize))
            for _ in range(comm_size)
        ]
        for _ in range(comm_size)
    ]
    for rank in range(0, comm_size):
        sources = [(rank - step) % comm_size for step in range(1, comm_size)]
        destination = [(rank + step) % comm_size for step in range(1, comm_size)]
        data_sizes_receive = [datasizes_randomized[src][rank] for src in sources]
        data_sizes_send = [datasizes_randomized[rank][dst] for dst in destination]

        parallel_send_recv(
            comm, rank, sources, destination, data_sizes_receive, data_sizes_send, tag
        )


def alltoall(algorithm, comm_size, **kwargs):
    comm = GoalComm(comm_size)
    if algorithm == "windowed":
        windowed_alltoall(comm, comm_size, **kwargs)
    elif algorithm == "balanced":
        balanced_alltoall(comm, comm_size, **kwargs)
    elif algorithm == "unbalanced":
        unbalanced_alltoall(comm, comm_size, **kwargs)
    else:
        raise ValueError(f"alltoall algorithm {algorithm} not implemented")
    return comm


def multi_alltoall(algorithm, num_comm_groups, comm_size, **kwargs):
    comm = GoalComm(comm_size * num_comm_groups)
    comms = comm.CommSplit(
        color=[i // comm_size for i in range(comm_size * num_comm_groups)],
        key=[i % comm_size for i in range(comm_size * num_comm_groups)],
    )
    for comm_split in comms:
        alltoall(algorithm, comm_split.CommSize(), **kwargs)
    return comm


