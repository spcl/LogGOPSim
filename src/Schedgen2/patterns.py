import random
from math import log2, ceil
from typing import List, Union
from goal import GoalComm

def binomialtree(comm_size: int, datasize: int, tag: int, algorithm: str = "reduce", ctd: int = 0, **kwargs) -> GoalComm:
    """
    Create a binomial tree communication pattern.

    :param comm_size: number of ranks in the communicator
    :param datasize: size of data to send or receive
    :param tag: tag that is used for all send and receive operations
    :param algorithm: communication algorithm that uses this pattern; default is reduce
    :param ctd: compute time dependency for each send operation; if 0 (default), no compute time is added
    :param kwargs: additional arguments that are ignored
    :return: GoalComm object that represents the communication pattern
    """
    assert algorithm in ["reduce", "bcast"], "direction must be reduce or bcast"
    comm = GoalComm(comm_size)
    for rank in range(0, comm_size):
        send = None
        recv = None
        for r in range(0, ceil(log2(comm_size))):
            peer = rank + pow(2, r)
            if (rank + pow(2, r) < comm_size) and (rank < pow(2, r)):
                if algorithm == "reduce":
                    recv = comm.Recv(size=datasize, src=peer, dst=rank, tag=tag)
                elif algorithm == "bcast":
                    send = comm.Send(size=datasize, dst=peer, src=rank, tag=tag)
                else:
                    raise ValueError(
                        "direction " + str(algorithm) + " in binomialtree not implemented."
                    )
            if (send is not None) and (recv is not None):
                if ctd > 0:
                    calc = comm.Calc(host=rank, size=ctd)
                    calc.requires(recv)
                    send.requires(calc)
                else:
                    send.requires(recv)
            peer = rank - pow(2, r)
            if (rank >= pow(2, r)) and (rank < pow(2, r + 1)):
                if algorithm == "reduce":
                    send = comm.Send(size=datasize, dst=peer, src=rank, tag=tag)
                if algorithm == "bcast":
                    recv = comm.Recv(size=datasize, src=peer, dst=rank, tag=tag)

    return comm

def recdoub(comm_size: int, datasize: int, tag: int, algorithm: str = "reduce-scatter", ctd: int = 0, **kwargs) -> GoalComm:
    """
    Create a recursive doubling communication pattern.

    :param comm_size: number of ranks in the communicator
    :param datasize: size of data to send or receive
    :param tag: tag that is used for all send and receive operations
    :param algorithm: communication algorithm that uses this pattern; default is reduce-scatter
    :param ctd: compute time dependency for each send operation; if 0 (default), no compute time is added
    :param kwargs: additional arguments that are ignored
    :return: GoalComm object that represents the communication pattern
    """

    assert algorithm in ["reduce-scatter", "allgather"], f"the pattern does not currently support the {algorithm} algorithm"

    comm = GoalComm(comm_size)
    num_steps = int(log2(comm_size))
    dependencies = [None] * comm_size
    for r in range(num_steps):
        for rank in range(comm_size):
            if algorithm in ["reduce-scatter"]:
                dest = rank ^ (2 ** r)
                message_size = datasize // (2 ** (r + 1))
            elif algorithm in ["allgather"]:
                dest = rank ^ (2 ** (num_steps - r - 1))
                message_size = datasize // (2 ** (num_steps - r))
            else:
                raise ValueError(f"the pattern does not currently support the {algorithm} algorithm")
            if dest < comm_size:
                send = comm.Send(size=message_size, src=rank, dst=dest, tag=tag + r)
                if dependencies[rank] is not None:
                    send.requires(dependencies[rank])
                dependencies[rank] = comm.Recv(size=message_size, src=dest, dst=rank, tag=tag + r)
                if ctd > 0:
                    calc = comm.Calc(host=rank, size=ctd)
                    calc.requires(dependencies[rank])
                    dependencies[rank] = calc
    return comm



def ring(comm_size: int, datasize: int, tag: int, rounds: int = 1, ctd: int = 0, **kwargs) -> GoalComm:
    """
    Create a ring communication pattern.

    :param comm_size: number of ranks in the communicator
    :param datasize: size of data to send in each round
    :param tag: base tag that is incremented for each round
    :param rounds: number of rounds to send data around the ring
    :param ctd: compute time dependency for each send operation; if 0 (default), no compute time is added
    :param kwargs: additional arguments that are ignored
    :return: GoalComm object that represents the communication pattern
    """
    comm = GoalComm(comm_size)
    dependencies = [None] * comm_size
    for r in range(rounds):
        for rank in range(comm_size):
            send = comm.Send(size=datasize, src=rank, dst=(rank + 1) % comm_size, tag=tag + r)
            if dependencies[rank] is not None:
                send.requires(dependencies[rank])
            dependencies[rank] = comm.Recv(size=datasize, src=(rank - 1) % comm_size, dst=rank, tag=tag + r)
            if ctd > 0:
                calc = comm.Calc(host=rank, size=ctd)
                calc.requires(dependencies[rank])
                dependencies[rank] = calc
    return comm

def _single_source_or_destination_linear(comm: GoalComm, anchor: int, datasizes: Union[List[int], List[List[int]]], tag: int, algorithm: str = "bcast", parallel: bool = True, window_size: int = 0, ctd: int = 0) -> GoalComm:
    """
    Create a single source or destination linear communication pattern.

    :param comm: GoalComm object that contains the ranks
    :param anchor: rank that is the source or destination
    :param datasizes: size(s) of data to send or receive
    :param tag: tag that is used for all send and receive operations
    :param algorithm: communication algorithm that uses this pattern; default is bcast (single source, multiple destinations)
    :param parallel: whether to send multiple messages in parallel; default is True (send messages in parallel)
    :param window_size: number of operations that can be in flight at once; default is 0 (no windowing)
    :param ctd: compute time dependency for each send operation; default is 0 (no compute time)
    :return: GoalComm object that represents the communication pattern
    """
    assert algorithm in ["bcast", "reduce", "alltoall", "incast", "outcast"], f"the pattern does not currently support the {algorithm} algorithm"
    assert (parallel and window_size == 0 and ctd == 0) or algorithm not in ["reduce", "incast"], f"We do not introduce dependencies, windowing, or compute time for linear receives"

    dependency = None
    if window_size > 0:
        window = [None] * window_size
        next_slot = 0
    for rank in range(comm.comm_size):
        if rank == anchor:
            continue
        if algorithm in ["bcast", "alltoall", "outcast"]:
            if algorithm == "alltoall":
                datasize = datasizes[anchor][rank]
            else:
                datasize = datasizes[rank]
            send = comm.Send(src=anchor, dst=rank, size=datasize, tag=tag)
            recv = comm.Recv(src=anchor, dst=rank, size=datasize, tag=tag)
            if not parallel:
                if window_size == 0:
                    if dependency is not None:
                        send.requires(dependency)
                    dependency = send
                    if ctd > 0:
                        calc = comm.Calc(host=anchor, size=ctd)
                        calc.requires(dependency)
                        dependency = calc
                else:
                    if window[next_slot] is not None:
                        send.requires(window[next_slot])
                    window[next_slot] = send
                    next_slot = (next_slot + 1) % window_size
                    if ctd > 0:
                        send.requires(comm.Calc(host=anchor, size=ctd))
            else:
                if ctd > 0:
                    send.requires(comm.Calc(host=anchor, size=ctd)) 
        elif algorithm in ["reduce", "incast"]:
            datasize = datasizes[rank]
            send = comm.Send(src=rank, dst=anchor, size=datasize, tag=tag)
            recv = comm.Recv(src=rank, dst=anchor, size=datasize, tag=tag)
            if ctd > 0:
                send.requires(comm.Calc(host=rank, size=ctd))
        else:
            raise ValueError(f"the pattern does not currently support the {algorithm} algorithm")

def linear(comm_size: int, datasize: int, tag: int, algorithm: str = "bcast", parallel: bool = True, randomized_data: bool = False, window_size: int = 0, ctd: int = 0, **kwargs) -> GoalComm:
    """
    Create a linear communication pattern.

    :param comm_size: number of ranks in the communicator
    :param datasize: size of data to send
    :param tag: tag that is used for all send and receive operations
    :param algorithm: communication algorithm that uses this pattern; default is bcast (single source, multiple destinations)
    :param parallel: whether to send multiple messages in parallel; default is True (send messages in parallel)
    :param randomized_data: whether to randomize the data sent or received; default is False (same size for all messages)
    :param window_size: number of operations that can be in flight at once; default is 0 (no windowing)
    :param ctd: compute time dependency for each send operation; default is 0 (no compute time)
    :param kwargs: additional arguments that are ignored
    :return: GoalComm object that represents the communication pattern
    """
    comm = GoalComm(comm_size)

    assert algorithm in ["bcast", "reduce", "alltoall", "incast", "outcast"], f"the pattern does not currently support the {algorithm} algorithm"

    if algorithm in ["alltoall"]:
        datasizes = [
            [
                (datasize + int(0.1 * random.randint(-datasize, datasize))) if randomized_data else datasize
                for _ in range(comm_size)
            ]
            for _ in range(comm_size)
        ]

        for anchor in range(comm_size):
            _single_source_or_destination_linear(comm, anchor, datasizes, tag, algorithm, parallel, window_size, ctd)
    else:
        datasizes = [
            (datasize + int(0.1 * random.randint(-datasize, datasize))) if randomized_data else datasize
            for _ in range(comm_size)
        ]
        _single_source_or_destination_linear(comm, 0, datasizes, tag, algorithm, parallel, window_size, ctd)

    return comm
    





#def _prepare_send_recv_data(
#    sources: Union[int, List[int]],
#    destinations: Union[int, List[int]],
#    data_sizes_receive: Union[int, List[int]],
#    data_sizes_send: Union[int, List[int]],
#    tags: Union[int, List[int]],
#):
#    if isinstance(sources, int) and isinstance(destinations, int):
#        sources = [sources]
#        destinations = [destinations]
#    elif isinstance(sources, int):
#        sources = [sources] * len(destinations)
#    elif isinstance(destinations, int):
#        destinations = [destinations] * len(sources)
#    assert len(sources) == len(
#        destinations
#    ), "sources and destinations must be the same length"
#    if isinstance(data_sizes_receive, int):
#        data_sizes_receive = [data_sizes_receive] * len(sources)
#    if isinstance(data_sizes_send, int):
#        data_sizes_send = [data_sizes_send] * len(destinations)
#    assert len(data_sizes_receive) == len(
#        sources
#    ), "data_sizes_receive and sources must be the same length"
#    assert len(data_sizes_send) == len(
#        destinations
#    ), "data_sizes_send and destinations must be the same length"
#    if isinstance(tags, int):
#        tags = [tags] * len(sources)
#    assert len(tags) == len(sources), "tags and sources must be the same length"
#    return sources, destinations, data_sizes_receive, data_sizes_send, tags
#
#
#def iterative_send_recv(
#    goal_comm: GoalComm,
#    rank: int,
#    sources: Union[int, List[int]],
#    destinations: Union[int, List[int]],
#    data_sizes_receive: Union[int, List[int]],
#    data_sizes_send: Union[int, List[int]],
#    tags: Union[int, List[int]],
#    last_dependency: GoalOp = None,
#    compute_time_dependency=0,
#) -> GoalOp:
#    """
#    Receive data from sources at rank and send data from rank to destinations with dependencies.
#
#    :param goal_comm: GoalComm object that contains the ranks
#    :param rank: rank to receive data at from sources and send data to destinations
#    :param sources: rank(s) to receive data from
#    :param destinations: rank(s) to send data to
#    :param data_sizes_receive: size(s) of data to receive from sources
#    :param data_sizes_send: size(s) of data to send to destinations
#    :param tags: tags to use for send and receive operations
#    :param last_dependency: last operation in a previous chain of operations to depend on
#    :param compute_time_dependency: time to compute before sending data.
#        If 0 (default), no compute time is added and the send operation is dependent on the receive operation.
#    :return: GoalOp object that represents the last operation in the chain
#    """
#    dependency = last_dependency
#
#    for source, destination, data_size_receive, data_size_send, tag in zip(
#        *_prepare_send_recv_data(
#            sources, destinations, data_sizes_receive, data_sizes_send, tags
#        )
#    ):
#        send = goal_comm.Send(src=rank, dst=destination, size=data_size_send, tag=tag)
#        if dependency is not None:
#            send.requires(dependency)
#        dependency = goal_comm.Recv(
#            src=source, dst=rank, size=data_size_receive, tag=tag
#        )
#        if compute_time_dependency > 0:
#            dependency = goal_comm.Calc(host=rank, size=compute_time_dependency)
#    return dependency
#
#
#def windowed_send_recv(
#    goal_comm: GoalComm,
#    rank: int,
#    sources: Union[int, List[int]],
#    destinations: Union[int, List[int]],
#    data_sizes_receive: Union[int, List[int]],
#    data_sizes_send: Union[int, List[int]],
#    window_size: int,
#    tags: Union[int, List[int]],
#    last_dependencies: List[GoalOp] = None,
#):
#    """
#    Receive data from sources at rank and send data from rank to destinations without dependencies.
#
#    :param goal_comm: GoalComm object that contains the ranks
#    :param rank: rank to receive data at from sources and send data to destinations
#    :param sources: rank(s) to receive data from
#    :param destinations: rank(s) to send data to
#    :param data_sizes_receive: size(s) of data to receive from sources
#    :param data_sizes_send: size(s) of data to send to destinations
#    :param window_size: number of operations that can be in flight at once
#    :param tags: tags to use for send and receive operations
#    :param last_dependencies: last operations in a previous chain of operations to depend on
#    """
#    assert (
#        not last_dependencies or len(last_dependencies) == window_size
#    ), "last_dependencies must be the same length as window_size"
#
#    window = last_dependencies or [None] * window_size
#
#    for i, (source, destination, data_size_receive, data_size_send, tag) in enumerate(
#        zip(
#            *_prepare_send_recv_data(
#                sources, destinations, data_sizes_receive, data_sizes_send, tags
#            )
#        )
#    ):
#        send = goal_comm.Send(src=rank, dst=destination, size=data_size_send, tag=tag)
#        if window[i % window_size] is not None:
#            send.requires(window[i % window_size])
#        window[i % window_size] = send
#        goal_comm.Recv(src=source, dst=rank, size=data_size_receive, tag=tag)
#
#    return window
#
#
#def parallel_send_recv(
#    goal_comm: GoalComm,
#    rank: int,
#    sources: Union[int, List[int]],
#    destinations: Union[int, List[int]],
#    data_sizes_receive: Union[int, List[int]],
#    data_sizes_send: Union[int, List[int]],
#    tags: Union[int, List[int]],
#):
#    """
#    Receive data from sources at rank and send data from rank to destinations without dependencies.
#
#    :param goal_comm: GoalComm object that contains the ranks
#    :param rank: rank to receive data at from sources and send data to destinations
#    :param sources: rank(s) to receive data from
#    :param destinations: rank(s) to send data to
#    :param data_sizes_receive: size(s) of data to receive from sources
#    :param data_sizes_send: size(s) of data to send to destinations
#    :param tags: tags to use for send and receive operations
#    """
#    for source, destination, data_size_receive, data_size_send, tag in zip(
#        *_prepare_send_recv_data(
#            sources, destinations, data_sizes_receive, data_sizes_send, tags
#        )
#    ):
#        goal_comm.Send(src=rank, dst=destination, size=data_size_send, tag=tag)
#        goal_comm.Recv(src=source, dst=rank, size=data_size_receive, tag=tag)
