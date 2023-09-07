import sys
import json
import tempfile
import subprocess
import argparse
from mpi_colls import *

parser = argparse.ArgumentParser(description="Generate GOAL Schedules.")

subparsers = parser.add_subparsers(
    help="Communication to generate", dest="comm", required=True
)
mpi = []

incast_parser = subparsers.add_parser("incast")
mpi.append(incast_parser)

outcast_parser = subparsers.add_parser("outcast")
mpi.append(outcast_parser)

dissemination_parser = subparsers.add_parser("dissemination")
mpi.append(dissemination_parser)

reduce_parser = subparsers.add_parser("reduce")
mpi.append(reduce_parser)

bcast_parser = subparsers.add_parser("bcast")
mpi.append(bcast_parser)

allreduce_parser = subparsers.add_parser("allreduce")
mpi.append(allreduce_parser)

alltoall_parser = subparsers.add_parser("alltoall")
mpi.append(alltoall_parser)

for p in ["incast", "outcast", "alltoall"]:
    p.add_argument(
        "--unbalanced",
        dest="unbalanced",
        action="store_true",
        help="Use unbalanced data sizes",
    )

for p in ["allreduce", "alltoall"]:
    p.add_argument(
        "--num_comm_groups",
        dest="num_comm_groups",
        type=int,
        default=1,
        help="Number of communication groups, >1 for multi-allreduce and multi-alltoall",
    )

for p in mpi:
    p.add_argument(
        "--ptrn",
        dest="ptrn",
        choices=[
            "datasize_based",
            "binomialtree",
            "recdoub",
            "ring",
            "linear"
        ],
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
        "--compute_dependency_time",
        dest="ctd",
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
    assert args.ptrn != "recdoub" or args.comm_size & (args.comm_size - 1) == 0, "Currently recdoub pattern requires a power of 2 communicator size."

def communication_pattern_selection(args):
    if args.ptrn_config is not None and args.ptrn == "datasize_based":
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
        with open(args.ptrn_config, "r") as f:
            config = json.load(f)
            comm_size = args.comm_size
            datasize = args.datasize
            for c in config:
                if "algorithm" in c and c["algorithm"] != "" and c["algorithm"] != args.comm:
                    continue
                if c["lower_bounds"]["comm_size"] != -1 and comm_size < c["lower_bounds"]["comm_size"]:
                    continue
                if c["upper_bounds"]["comm_size"] != -1 and comm_size >= c["upper_bounds"]["comm_size"]:
                    continue
                if c["lower_bounds"]["datasize"] != -1 and datasize < c["lower_bounds"]["datasize"]:
                    continue
                if c["upper_bounds"]["datasize"] != -1 and datasize >= c["upper_bounds"]["datasize"]:
                    continue
                if c["lower_bounds"]["combined"] is not None:
                    for grad, intercept in c["lower_bounds"]["combined"]:
                        if datasize < grad * comm_size + intercept:
                            continue
                if c["upper_bounds"]["combined"] is not None:
                    for grad, intercept in c["upper_bounds"]["combined"]:
                        if datasize >= grad * comm_size + intercept:
                            continue
                args.ptrn = c["ptrn"]
                break
            if args.ptrn == "datasize_based":
                raise ValueError(f"Cannot find a pattern for comm_size={comm_size} and datasize={datasize} according to the config file")
    elif args.ptrn == "datasize_based":
        if args.comm == "incast":
            args.ptrn = "linear"
        elif args.comm == "outcast":
            args.ptrn = "linear"
        elif args.comm == "reduce":
            # use binomial tree for large data size and when the communicator size is a power of 2
            if args.datasize > 4096 and args.comm_size & (args.comm_size - 1) == 0:
                args.ptrn = "binomialtree"
            else:
                args.ptrn = "linear"
        elif args.comm == "bcast":
            # use binomial tree for small data size and when the communicator size is a power of 2
            if args.datasize <= 4096 and args.comm_size & (args.comm_size - 1) == 0:
                args.ptrn = "binomialtree"
            else:
                args.ptrn = "linear"
        elif args.comm == "dissemination":
            # TODO currently not implemented to support different patterns
            pass
        elif args.comm == "allreduce":
            # Use recdoub for power of 2 communicator size and small data sizes
            if args.datasize <= 4096 and args.comm_size & (args.comm_size - 1) == 0:
                args.ptrn = "recdoub"
            else:
                args.ptrn = "ring"
        elif args.comm == "alltoall":
            args.ptrn = "linear"
        else:
            raise ValueError(f"Communication type {args.comm} not implemented")
    else:
        # Pattern was specified by the user
        pass

args = parser.parse_args()
if args.config is not None:
    with open(args.config, "r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)

communication_pattern_selection(args)
verify_params(args)
args.tag = 42

if args.comm == "incast":
    g = incast(**vars(args))
elif args.comm == "outcast":
    g = outcast(**vars(args))
elif args.comm == "reduce":
    g = reduce(**vars(args))
elif args.comm == "bcast":
    g = bcast(**vars(args))
elif args.comm == "dissemination":
    g = dissemination(**vars(args))
elif args.comm == "allreduce":
    g = allreduce(**vars(args))
elif args.comm == "alltoall":
    g = alltoall(**vars(args))

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
