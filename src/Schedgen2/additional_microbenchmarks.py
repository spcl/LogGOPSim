from patterns import linear


def incast(
    comm_size: int,
    datasize: int,
    tag: int = 42,
    ptrn: str = "linear",
    randomized_data: bool = False,
    **kwargs,
):
    assert ptrn == "linear", "incast only supports the linear communication pattern"
    return linear(
        comm_size=comm_size,
        datasize=datasize,
        tag=tag,
        algorithm="incast",
        parallel=True,
        randomized_data=randomized_data,
        **kwargs,
    )


def outcast(
    comm_size: int,
    datasize: int,
    tag: int = 42,
    ptrn: str = "linear",
    randomized_data: bool = False,
    **kwargs,
):
    assert ptrn == "linear", "outcast only supports the linear communication pattern"
    return linear(
        comm_size=comm_size,
        datasize=datasize,
        tag=tag,
        algorithm="outcast",
        parallel=True,
        randomized_data=randomized_data,
        **kwargs,
    )
