from typing import List, Union
from goal import GoalComm, GoalOp


def _prepare_send_recv_data(
    sources: Union[int, List[int]],
    destinations: Union[int, List[int]],
    data_sizes_receive: Union[int, List[int]],
    data_sizes_send: Union[int, List[int]],
):
    if isinstance(sources, int) and isinstance(destinations, int):
        sources = [sources]
        destinations = [destinations]
    elif isinstance(sources, int):
        sources = [sources] * len(destinations)
    elif isinstance(destinations, int):
        destinations = [destinations] * len(sources)
    assert len(sources) == len(
        destinations
    ), "sources and destinations must be the same length"
    if isinstance(data_sizes_receive, int):
        data_sizes_receive = [data_sizes_receive] * len(sources)
    if isinstance(data_sizes_send, int):
        data_sizes_send = [data_sizes_send] * len(destinations)
    assert len(data_sizes_receive) == len(
        sources
    ), "data_sizes_receive and sources must be the same length"
    assert len(data_sizes_send) == len(
        destinations
    ), "data_sizes_send and destinations must be the same length"
    return sources, destinations, data_sizes_receive, data_sizes_send


def iterative_send_recv(
    goal_comm: GoalComm,
    rank: int,
    sources: Union[int, List[int]],
    destinations: Union[int, List[int]],
    data_sizes_receive: Union[int, List[int]],
    data_sizes_send: Union[int, List[int]],
    tag,
    last_dependency: GoalOp = None,
    compute_time_dependency=0,
) -> GoalOp:
    """
    Receive data from sources at rank and send data from rank to destinations with dependencies.

    :param goal_comm: GoalComm object that contains the ranks
    :param rank: rank to receive data at from sources and send data to destinations
    :param sources: rank(s) to receive data from
    :param destinations: rank(s) to send data to
    :param data_sizes_receive: size(s) of data to receive from sources
    :param data_sizes_send: size(s) of data to send to destinations
    :param tag: tag to use for send and receive operations
    :param last_dependency: last operation in a previous chain of operations to depend on
    :param compute_time_dependency: time to compute before sending data.
        If 0 (default), no compute time is added and the send operation is dependent on the receive operation.
    :return: GoalOp object that represents the last operation in the chain
    """
    dependency = last_dependency

    for source, destination, data_size_receive, data_size_send in zip(
        *_prepare_send_recv_data(
            sources, destinations, data_sizes_receive, data_sizes_send
        )
    ):
        send = goal_comm.Send(src=rank, dst=destination, size=data_size_send, tag=tag)
        if dependency is not None:
            send.requires(dependency)
        dependency = goal_comm.Recv(
            src=source, dst=rank, size=data_size_receive, tag=tag
        )
        if compute_time_dependency > 0:
            dependency = goal_comm.Calc(host=rank, size=compute_time_dependency)
    return dependency


def windowed_send_recv(
    goal_comm: GoalComm,
    rank: int,
    sources: Union[int, List[int]],
    destinations: Union[int, List[int]],
    data_sizes_receive: Union[int, List[int]],
    data_sizes_send: Union[int, List[int]],
    window_size: int,
    tag,
    last_dependencies: List[GoalOp] = None,
):
    """
    Receive data from sources at rank and send data from rank to destinations without dependencies.

    :param goal_comm: GoalComm object that contains the ranks
    :param rank: rank to receive data at from sources and send data to destinations
    :param sources: rank(s) to receive data from
    :param destinations: rank(s) to send data to
    :param data_sizes_receive: size(s) of data to receive from sources
    :param data_sizes_send: size(s) of data to send to destinations
    :param window_size: number of operations that can be in flight at once
    :param tag: tag to use for send and receive operations
    :param last_dependencies: last operations in a previous chain of operations to depend on
    """
    assert (
        not last_dependencies or len(last_dependencies) == window_size
    ), "last_dependencies must be the same length as window_size"

    window = last_dependencies or [None] * window_size

    for i, (source, destination, data_size_receive, data_size_send) in enumerate(
        zip(
            *_prepare_send_recv_data(
                sources, destinations, data_sizes_receive, data_sizes_send
            )
        )
    ):
        send = goal_comm.Send(src=rank, dst=destination, size=data_size_send, tag=tag)
        if window[i % window_size] is not None:
            send.requires(window[i % window_size])
        window[i % window_size] = send
        goal_comm.Recv(src=source, dst=rank, size=data_size_receive, tag=tag)

    return window


def parallel_send_recv(
    goal_comm: GoalComm,
    rank: int,
    sources: Union[int, List[int]],
    destinations: Union[int, List[int]],
    data_sizes_receive: Union[int, List[int]],
    data_sizes_send: Union[int, List[int]],
    tag,
):
    """
    Receive data from sources at rank and send data from rank to destinations without dependencies.

    :param goal_comm: GoalComm object that contains the ranks
    :param rank: rank to receive data at from sources and send data to destinations
    :param sources: rank(s) to receive data from
    :param destinations: rank(s) to send data to
    :param data_sizes_receive: size(s) of data to receive from sources
    :param data_sizes_send: size(s) of data to send to destinations
    :param tag: tag to use for send and receive operations
    """
    for source, destination, data_size_receive, data_size_send in zip(
        *_prepare_send_recv_data(
            sources, destinations, data_sizes_receive, data_sizes_send
        )
    ):
        goal_comm.Send(src=rank, dst=destination, size=data_size_send, tag=tag)
        goal_comm.Recv(src=source, dst=rank, size=data_size_receive, tag=tag)
