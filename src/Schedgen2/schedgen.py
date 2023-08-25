import sys
import argparse
from patterns import binomialtree, dissemination, ring_allreduce

parser = argparse.ArgumentParser(description="Generate GOAL Schedules.")
parser.add_argument(
    "--ptrn",
    dest="ptrn",
    choices=[
        "binomialtreereduce",
        "binarytreebcast",
        "dissemination",
        "ring_allreduce",
    ],
    help="Pattern to generate",
    required=True,
)
parser.add_argument(
    "--commsize", dest="commsize", type=int, default=8, help="Size of the communicator"
)
parser.add_argument(
    "--datasize",
    dest="datasize",
    type=int,
    default=8,
    help="Size of the data, i.e., for reduce operations",
)
parser.add_argument("--output", dest="output", default="stdout", help="Output file")
parser.add_argument(
    "--ignore_verification",
    dest="ignore_verification",
    action="store_true",
    help="Ignore verification of parameters",
)


def verify_params(args):
    if args.ignore_verification:
        return
    assert args.commsize > 0, "Communicator size must be greater than 0."
    assert args.datasize > 0, "Data size must be greater than 0."


args = parser.parse_args()

verify_params(args)

if args.output == "stdout":
    args.output = sys.stdout
else:
    args.output = open(args.output, "w")

if args.ptrn == "binomialtreereduce":
    g = binomialtree(args.commsize, args.datasize, 42, "reduce")
elif args.ptrn == "binomialtreebcast":
    g = binomialtree(args.commsize, args.datasize, 42, "bcast")
elif args.ptrn == "dissemination":
    g = dissemination(args.commsize, args.datasize, 42)
elif args.ptrn == "ring_allreduce":
    g = ring_allreduce(args.commsize, args.datasize, 42)

g.write_goal(fh=args.output)
if args.output != sys.stdout:
    args.output.close()
