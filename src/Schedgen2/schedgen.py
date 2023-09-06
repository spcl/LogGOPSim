import sys
import json
import tempfile
import subprocess
import argparse
from mpi_colls import *

parser = argparse.ArgumentParser(description="Generate GOAL Schedules.")

subparsers = parser.add_subparsers(
    help="Pattern to generate", dest="ptrn", required=True
)
simple_patterns = []
multi_patterns = []

incast_parser = subparsers.add_parser("incast")
simple_patterns.append(incast_parser)

outcast_parser = subparsers.add_parser("outcast")
simple_patterns.append(outcast_parser)

for p in [incast_parser, outcast_parser]:
    p.add_argument(
        "--unbalanced",
        dest="unbalanced",
        action="store_true",
        help="Use unbalanced incast/outcast",
    )
binomialtreereduce_parser = subparsers.add_parser("binomialtreereduce")
simple_patterns.append(binomialtreereduce_parser)

binomialtreebcast_parser = subparsers.add_parser("binomialtreebcast")
simple_patterns.append(binomialtreebcast_parser)

dissemination_parser = subparsers.add_parser("dissemination")
simple_patterns.append(dissemination_parser)

allreduce_parser = subparsers.add_parser("allreduce")
simple_patterns.append(allreduce_parser)

multi_allreduce_parser = subparsers.add_parser("multi_allreduce")
multi_patterns.append(multi_allreduce_parser)

alltoall_parser = subparsers.add_parser("alltoall")
simple_patterns.append(alltoall_parser)

multi_alltoall_parser = subparsers.add_parser("multi_alltoall")
multi_patterns.append(multi_alltoall_parser)

for p in [
    allreduce_parser,
    multi_allreduce_parser,
]:
    p.add_argument(
        "--algorithm",
        dest="algorithm",
        choices=[
            "ring",
            "recdoub",
            "datasize_based",
        ],
        default="datasize_based",
        help="Algorithm to use for allreduce",
    )

for p in [
    alltoall_parser,
    multi_alltoall_parser,
]:
    p.add_argument(
        "--algorithm",
        dest="algorithm",
        choices=[
            "windowed",
            "balanced",
            "unbalanced",
        ],
        default="datasize_based",
        help="Algorithm to use for alltoall",
    )
    p.add_argument(
        "--window_size",
        dest="window_size",
        type=int,
        default=8,
        help="Window size for windowed alltoall",
    )

for p in simple_patterns + multi_patterns:
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
    p.add_argument("--output", dest="output", default="stdout", help="Output file")
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

for p in multi_patterns:
    p.add_argument(
        "--num_comm_groups",
        dest="num_comm_groups",
        type=int,
        required=True,
        help="Number of communication groups",
    )


def verify_params(args):
    if args.ignore_verification:
        return
    assert args.comm_size > 0, "Communicator size must be greater than 0."
    assert args.datasize > 0, "Data size must be greater than 0."
    assert (
        args.txt2bin is None or args.output != "stdout"
    ), "Cannot use txt2bin with stdout"


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
