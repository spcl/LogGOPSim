import json
from goal import GoalComm
from patterns import binomialtree, recdoub, ring, linear


def mpi_communication_pattern_selection(
    algorithm: str, comm_size: int, datasize: int, ptrn_config: str = None
):
    if ptrn_config is not None and ptrn_config != "":
        # The config file should be a json file with the following format (lower bounds are inclusive, upper bounds are exclusive):
        # [
        #     {
        #         "algorithm": "algorithm_name", # can be left empty or omitted, otherwise only matching algorithms are considered
        #         "ptrn": "pattern_name",
        #         "lower_bounds": {
        #             "comm_size": -1 for no lower bound on the x-axis,
        #             "datasize": -1 for no lower bound on the y-axis,
        #             "combined": [(grad, intercept), (grad, intercept), ...] for the combined lower bounds
        #         },
        #         "upper_bounds": {
        #             "comm_size": -1 for no upper bound on the x-axis,
        #             "datasize": -1 for no upper bound on the y-axis,
        #             "combined": [(grad, intercept), (grad, intercept), ...] for the combined upper bounds
        #         }
        #     },
        #     ...
        # ]
        with open(ptrn_config, "r") as f:
            config = json.load(f)
            for c in config:
                if (
                    "algorithm" in c
                    and c["algorithm"] != ""
                    and c["algorithm"] != algorithm
                ):
                    continue
                if (
                    c["lower_bounds"]["comm_size"] != -1
                    and comm_size < c["lower_bounds"]["comm_size"]
                ):
                    continue
                if (
                    c["upper_bounds"]["comm_size"] != -1
                    and comm_size >= c["upper_bounds"]["comm_size"]
                ):
                    continue
                if (
                    c["lower_bounds"]["datasize"] != -1
                    and datasize < c["lower_bounds"]["datasize"]
                ):
                    continue
                if (
                    c["upper_bounds"]["datasize"] != -1
                    and datasize >= c["upper_bounds"]["datasize"]
                ):
                    continue
                if c["lower_bounds"]["combined"] is not None:
                    for grad, intercept in c["lower_bounds"]["combined"]:
                        if datasize < grad * comm_size + intercept:
                            continue
                if c["upper_bounds"]["combined"] is not None:
                    for grad, intercept in c["upper_bounds"]["combined"]:
                        if datasize >= grad * comm_size + intercept:
                            continue
                return c["ptrn"]
            raise ValueError(
                f"Cannot find a pattern for comm_size={comm_size} and datasize={datasize} according to the config file"
            )
    else:
        if algorithm == "reduce":
            # use binomial tree for large data size and when the communicator size is a power of 2
            if datasize > 4096 and comm_size & (comm_size - 1) == 0:
                return "binomialtree"
            else:
                return "linear"
        elif algorithm == "bcast":
            # use binomial tree for small data size and when the communicator size is a power of 2
            if datasize <= 4096 and comm_size & (comm_size - 1) == 0:
                return "binomialtree"
            else:
                return "linear"
        elif algorithm == "dissemination":
            # TODO currently not implemented to support different patterns
            pass
        elif algorithm == "allreduce":
            # Use recdoub for power of 2 communicator size and small data sizes
            if datasize <= 4096 and comm_size & (comm_size - 1) == 0:
                return "recdoub"
            else:
                return "ring"
        elif algorithm == "alltoall" or algorithm == "alltoallv":
            return "linear"
        else:
            raise ValueError(f"Communication type {algorithm} not implemented")


def dissemination(comm_size, datasize, tag):
    # TODO: select or implement right pattern
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


def scatter(
    comm_size: int,
    datasize: int,
    tag: int = 42,
    ptrn: str = "linear",
    **kwargs,
):
    if ptrn == "binomialtree":
        return binomialtree(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="scatter",
            **kwargs,
        )
    elif ptrn == "linear":
        return linear(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="scatter",
            parallel=True,
            **kwargs,
        )
    else:
        raise ValueError(f"scatter with pattern {ptrn} not implemented")


def reduce(
    comm_size: int,
    datasize: int,
    tag: int = 42,
    ptrn: str = "binomialtree",
    **kwargs,
):
    if ptrn == "binomialtree":
        return binomialtree(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="reduce",
            **kwargs,
        )
    elif ptrn == "linear":
        return linear(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="reduce",
            parallel=True,
            **kwargs,
        )
    else:
        raise ValueError(f"reduce with pattern {ptrn} not implemented")


def bcast(
    comm_size: int,
    datasize: int,
    tag: int = 42,
    ptrn: str = "binomialtree",
    **kwargs,
):
    if ptrn == "binomialtree":
        return binomialtree(
            comm_size=comm_size, datasize=datasize, tag=tag, algorithm="bcast", **kwargs
        )
    elif ptrn == "linear":
        return linear(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="bcast",
            parallel=True,
            **kwargs,
        )
    else:
        raise ValueError(f"bcast with pattern {ptrn} not implemented")


def allreduce(
    comm_size: int,
    datasize: int,
    tag: int = 42,
    ptrn: str = "recdoub",
    **kwargs,
):
    comms = []  # reduce-scatter and allgather
    if ptrn == "recdoub":
        comms.append(
            recdoub(
                comm_size=comm_size,
                datasize=datasize,
                tag=tag,
                algorithm="reduce-scatter",
                **kwargs,
            )
        )
        comms.append(
            recdoub(
                comm_size=comm_size,
                datasize=datasize,
                tag=tag + comm_size,
                algorithm="allgather",
                **kwargs,
            )
        )
    elif ptrn == "ring":
        comms.append(
            ring(
                comm_size=comm_size,
                datasize=datasize,
                tag=tag,
                algorithm="reduce-scatter",
                rounds=comm_size - 1,
                **kwargs,
            )
        )
        comms.append(
            ring(
                comm_size=comm_size,
                datasize=datasize,
                tag=tag + comm_size,
                algorithm="allgather",
                rounds=comm_size - 1,
                **kwargs,
            )
        )
    else:
        raise ValueError(f"allreduce with pattern {ptrn} not implemented")
    comms[0].Append(comms[1])
    return comms[0]


def alltoall(
    comm_size: int,
    datasize: int,
    tag: int = 42,
    ptrn: str = "linear",
    window_size: int = 0,
    **kwargs,
):
    if ptrn == "linear":
        return linear(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="alltoall",
            parallel=(window_size == 0),
            window_size=window_size,
            **kwargs,
        )
    else:
        raise ValueError(f"alltoall with pattern {ptrn} not implemented")


def alltoallv(
    comm_size: int,
    datasize: int,
    tag: int = 42,
    ptrn: str = "linear",
    window_size: int = 0,
    **kwargs,
):
    # TODO: currently data is only randomized, add support for custom data sizes
    if ptrn == "linear":
        return linear(
            comm_size=comm_size,
            datasize=datasize,
            tag=tag,
            algorithm="alltoallv",
            parallel=(window_size == 0),
            randomized_data=True,
            window_size=window_size,
            **kwargs,
        )
    else:
        raise ValueError(f"alltoallv with pattern {ptrn} not implemented")
