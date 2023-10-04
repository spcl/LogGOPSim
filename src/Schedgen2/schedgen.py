#! /usr/bin/env python3

import sys
import json
import tempfile
import subprocess
import argparse
from mpi_colls import *
from additional_microbenchmarks import *

parser = argparse.ArgumentParser(description="Generate GOAL Schedules.")

subparsers = parser.add_subparsers(
    help="Communication to generate", dest="comm", required=True
)
mpi = []
additional_microbenchmarks = []

incast_parser = subparsers.add_parser("incast")
additional_microbenchmarks.append(incast_parser)

outcast_parser = subparsers.add_parser("outcast")
additional_microbenchmarks.append(outcast_parser)

dissemination_parser = subparsers.add_parser("dissemination")
mpi.append(dissemination_parser)

reduce_parser = subparsers.add_parser("reduce")
mpi.append(reduce_parser)

bcast_parser = subparsers.add_parser("bcast")
mpi.append(bcast_parser)

scatter_parser = subparsers.add_parser("scatter")
mpi.append(scatter_parser)

allreduce_parser = subparsers.add_parser("allreduce")
mpi.append(allreduce_parser)

alltoall_parser = subparsers.add_parser("alltoall")
mpi.append(alltoall_parser)

alltoallv_parser = subparsers.add_parser("alltoallv")
mpi.append(alltoallv_parser)

for p in additional_microbenchmarks:
    p.add_argument(
        "--randomized_data",
        dest="randomized_data",
        action="store_true",
        help="Use unbalanced data sizes",
    )

for p in [allreduce_parser, alltoall_parser, alltoallv_parser]:
    p.add_argument(
        "--num_comm_groups",
        dest="num_comm_groups",
        type=int,
        default=1,
        help="Number of communication groups, >1 for multi-allreduce and multi-alltoall(v)",
    )

for p in mpi + additional_microbenchmarks:
    p.add_argument(
        "--ptrn",
        dest="ptrn",
        choices=["datasize_based", "binomialtree", "recdoub", "ring", "linear"],
        default="datasize_based",
        help="Pattern to use for communication, note that not all patterns are available for all communication types",
    )
    p.add_argument(
        "--ptrn-config",
        dest="ptrn_config",
        help="Configuration file for the pattern to use with data size based selection to override the default configuration",
    )
    p.add_argument(
        "--comm_size",
        dest="comm_size",
        type=int,
        default=8,
        help="Size of the communicator",
    )
    p.add_argument(
        "--datasize",
        dest="datasize",
        type=int,
        default=8,
        help="Size of the data, i.e., for reduce operations",
    )
    p.add_argument(
        "--window_size",
        dest="window_size",
        type=int,
        default=0,
        help="Window size for windowed linear communication patterns",
    )
    p.add_argument(
        "--compute_time_dependency",
        dest="compute_time_dependency",
        type=int,
        default=0,
        help="Compute time that is to be inserted in between send operations",
    )
    p.add_argument(
        "--output",
        dest="output",
        default="stdout",
        help="Output file",
    )
    p.add_argument(
        "--ignore_verification",
        dest="ignore_verification",
        action="store_true",
        help="Ignore verification of parameters",
    )
    p.add_argument(
        "--config",
        dest="config",
        help="Configuration file, takes precedence over other parameters",
    )
    p.add_argument(
        "--txt2bin",
        dest="txt2bin",
        help="Path to txt2bin executable",
    )


def verify_params(args):
    if args.ignore_verification:
        return
    assert args.comm_size > 0, "Communicator size must be greater than 0."
    assert args.datasize > 0, "Data size must be greater than 0."
    assert (
        args.txt2bin is None or args.output != "stdout"
    ), "Cannot use txt2bin with stdout"
    assert (
        args.ptrn != "recdoub" or args.comm_size & (args.comm_size - 1) == 0
    ), "Currently recdoub pattern requires a power of 2 communicator size."


def comm_to_func(comm: str) -> callable:
    """
    Convert a communication type to a function that generates the communication.

    :param comm: The communication type.
    :return: A function that generates the communication.
    """

    if comm == "incast":
        return incast
    elif comm == "outcast":
        return outcast
    elif comm == "reduce":
        return reduce
    elif comm == "bcast":
        return bcast
    elif comm == "scatter":
        return scatter
    elif comm == "dissemination":
        return dissemination
    elif comm == "allreduce":
        return allreduce
    elif comm == "alltoall":
        return alltoall
    elif comm == "alltoallv":
        return alltoallv
    else:
        raise ValueError(f"Communication type {comm} not implemented")


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


args = parser.parse_args()
if args.config is not None:
    with open(args.config, "r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)

if args.ptrn == "datasize_based":
    if args.comm in [p.prog.split()[-1] for p in mpi]:
        args.ptrn = mpi_communication_pattern_selection(
            args.comm, args.comm_size, args.datasize
        )
    elif args.comm in [p.prog.split()[-1] for p in additional_microbenchmarks]:
        args.ptrn = "linear"
    else:
        raise ValueError(
            f"Communication type {args.comm} does not currently support data size based pattern selection"
        )

verify_params(args)
args.tag = 42

if (
    "num_comm_groups" not in vars(args)
    or args.num_comm_groups is None
    or args.num_comm_groups <= 1
):
    g = comm_to_func(args.comm)(**vars(args))
else:
    g = multi(
        comm_to_func(args.comm), **vars(args)
    )

if args.txt2bin is not None:
    assert args.output != "stdout", "Cannot use txt2bin with stdout"
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        g.write_goal(fh=f)
        tmp_goal_file = f.name
    subprocess.run(
        [args.txt2bin, "-i", tmp_goal_file, "-o", args.output, "-p"],
        check=True,
    )
    subprocess.run(["rm", tmp_goal_file], check=True)
else:
    if args.output == "stdout":
        args.output = sys.stdout
    else:
        args.output = open(args.output, "w")

    g.write_goal(fh=args.output)
    if args.output != sys.stdout:
        args.output.close()
