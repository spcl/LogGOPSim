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

reduce_parser = subparsers.add_parser("reduce")
mpi.append(reduce_parser)

bcast_parser = subparsers.add_parser("bcast")
mpi.append(bcast_parser)

dissemination_parser = subparsers.add_parser("dissemination")
mpi.append(dissemination_parser)

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
        required=True,
        help="Number of communication groups",
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

def communication_pattern_selection(args):
    # check if 


args = parser.parse_args()
if args.config is not None:
    with open(args.config, "r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)


verify_params(args)

if args.ptrn == "binomialtreereduce":
    g = binomialtree(args.comm_size, args.datasize, 42, "reduce")
elif args.ptrn == "binomialtreebcast":
    g = binomialtree(args.comm_size, args.datasize, 42, "bcast")
elif args.ptrn == "dissemination":
    g = dissemination(args.comm_size, args.datasize, 42)
elif args.ptrn == "allreduce":
    g = allreduce(tag=42, **vars(args))
elif args.ptrn == "multi_allreduce":
    g = multi_allreduce(tag=42, **vars(args))
elif args.ptrn == "alltoall":
    g = alltoall(tag=42, **vars(args))
elif args.ptrn == "multi_alltoall":
    g = multi_alltoall(tag=42, **vars(args))
elif args.ptrn == "incast":
    g = incast(tag=42, **vars(args))
elif args.ptrn == "outcast":
    g = outcast(tag=42, **vars(args))

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
