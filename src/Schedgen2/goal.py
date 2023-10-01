import sys


class GoalLabeller:
    def __init__(self):
        self.next_label = 1
        self.next_comm = 1
        self.op_dict = {}
        self.comm_dict = {}

    def GetLabel(self, op):
        if op in self.op_dict:
            pass
        else:
            self.op_dict[op] = self.next_label
            self.next_label += 1
        return self.op_dict[op]

    def GetCommID(self, comm):
        if comm in self.comm_dict:
            pass
        else:
            self.comm_dict[comm] = self.next_comm
            self.next_comm += 1
        return self.comm_dict[comm]

    def MakeTag(self, tag, comm):
        """Combine the user tag and the comm tag portion"""
        return tag * 1000 + comm


class GoalOp:
    def __init__(self):
        self.depends_on = []

    def requires(self, required):
        # TODO check that self and required translate to the same rank in comm_world - we don't have the rank here :(
        self.depends_on.append(required)


class GoalSend(GoalOp):
    def __init__(self, dst, tag, size):
        super().__init__()
        self.dst = dst
        self.tag = tag
        self.size = size

    def write_goal(self, labeller, fh, comm, basecomm, format="goal"):
        if format == "goal":
            fh.write(
                "l{label}: send {size}b to {dst} tag {tag}\n".format(
                    label=labeller.GetLabel(self),
                    size=str(self.size),
                    dst=str(comm.TranslateRank(self.dst, basecomm)),
                    tag=str(labeller.MakeTag(self.tag, labeller.GetCommID(comm))),
                )
            )
        elif format == "graphviz":
            fh.write(
                "\"l{label}\" [label=\"send {size}b to {dst} tag {tag}\"];\n".format(
                    label=labeller.GetLabel(self),
                    size=str(self.size),
                    dst=str(comm.TranslateRank(self.dst, basecomm)),
                    tag=str(labeller.MakeTag(self.tag, labeller.GetCommID(comm))),
                )
            )
        else:
            raise NotImplementedError("Requested output format "+str(format)+" not implemented!")
        


class GoalRecv(GoalOp):
    def __init__(self, src, tag, size):
        super().__init__()
        self.src = src
        self.tag = tag
        self.size = size

    def write_goal(self, labeller, fh, comm, basecomm, format="goal"):
        if format == "goal":
            fh.write(
                "l{label}: recv {size}b from {src} tag {tag}\n".format(
                    label=labeller.GetLabel(self),
                    size=str(self.size),
                    src=str(comm.TranslateRank(self.src, basecomm)),
                    tag=str(labeller.MakeTag(self.tag, labeller.GetCommID(comm))),
                )
            )
        elif format == "graphviz":
            fh.write(
                "\"l{label}\" [label=\"recv {size}b from {src} tag {tag}\"];\n".format(
                    label=labeller.GetLabel(self),
                    size=str(self.size),
                    src=str(comm.TranslateRank(self.src, basecomm)),
                    tag=str(labeller.MakeTag(self.tag, labeller.GetCommID(comm))),
                )
            )
        else:
            raise NotImplementedError("Requested output format "+str(format)+" not implemented!")


class GoalCalc(GoalOp):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def write_goal(self, labeller, fh, comm, basecomm, format="goal"):
        if format == "goal":
            fh.write(
                "l{label}: calc {size}\n".format(
                    label=labeller.GetLabel(self), size=str(self.size))
            )
        elif format == "graphviz":
            fh.write(
                "\"l{label}\" [label=\"calc {size}\"]\n".format(
                    label=labeller.GetLabel(self), size=str(self.size)
                )
            )
        else:
            raise NotImplementedError("Requested output format "+str(format)+" not implemented!")

class GoalRank:
    def __init__(self, comm, rank):
        self.comm = comm
        self.rank = rank
        self.base_rank = None
        self.ops = []

    def Send(self, dst, tag, size):
        if dst > self.comm.CommSize():
            raise ValueError(str(dst) + " is larger than comm size!")
        op = GoalSend(dst=dst, tag=tag, size=size)
        self.ops.append(op)
        return op

    def Recv(self, src, tag, size):
        if src > self.comm.CommSize():
            raise ValueError(str(src) + " is larger than comm size!")
        op = GoalRecv(src=src, tag=tag, size=size)
        self.ops.append(op)
        return op

    def Calc(self, size):
        op = GoalCalc(size=size)
        self.ops.append(op)
        return op

    def Merge(self, mrank):
        self.ops += mrank.ops

    def Append(self, arank, dependOn=None, allOpsDepend=False):
        """ Append arank to self. If dependOn is None, all ops in self need to finish before we start executing aranks ops. If dependOn is given we only depend on that. 
            By default (allOpsDepend) only independent ops in arank depend on self, however if allOpsDepend=True all ops do. """
        if dependOn is None:
            c = self.Calc(0)
            for l in self.LastOps():
                if l == c:
                    pass
                else:
                    c.requires(l)
        else:
            c = dependOn
        self.ops += arank.ops
        depops = arank.IndepOps()
        if allOpsDepend:
            depops = arank.ops
        for i in depops:
            i.requires(c) 

    def IndepOps(self):
        res = [x for x in self.ops if (len(x.depends_on) == 0)]
        return res

    def LastOps(self):
        rem = []
        for x in self.ops:
            for d in x.depends_on:
                rem.append(d)
        s = set(rem)
        res = [x for x in self.ops if x not in s]
        return res


    def write_goal(self, labeller, fh, rankid=True, basecomm=None, format="goal"):
        if basecomm is None:
            basecomm = (
                self.comm
            )  # stupid python evals default args at method definition, not call time :(
        if rankid:
            if format == "goal":
                fh.write("rank " + str(self.rank) + " {\n")
            elif format == "graphviz":
                fh.write("subgraph cluster_" + str(self.rank) + " {\n")
                fh.write("style=filled; color=lightgrey; node [style=filled,color=white]; label=\"rank "+str(self.rank)+"\";")
        for op in self.ops:
            op.write_goal(labeller, fh, self.comm, basecomm, format=format)
        for op in self.ops:
            for req in op.depends_on:
                if format == "goal":
                    fh.write(
                        "l{label1} requires l{label2}\n".format(
                            label1=labeller.GetLabel(op), label2=labeller.GetLabel(req)
                        )
                    )
                if format == "graphviz":
                    # we "invert" dependencies in grphviz format, i.e, a->b means a is executed before b.
                    # Where in goal it would be "b requires a" - but this would make graphs look upside down. 
                    fh.write(
                        "l{label2} -> l{label1}\n".format(
                            label1=labeller.GetLabel(op), label2=labeller.GetLabel(req)
                        )
                    )
        for sc in self.comm.subcomms:
            sc.write_goal_subcomm(labeller, fh, self.rank, basecomm, format=format)
        if rankid:
            fh.write("}\n\n")


class GoalComm:
    def __init__(self, comm_size):
        self.base_comm = self
        self.comm_size = comm_size
        self.subcomms = []
        self.ranks = [GoalRank(comm=self, rank=rank) for rank in range(comm_size)]

    def __getitem__(self, index):
        return self.ranks[index]

    def Append(self, comm):
        """Append comm to self, such that when all ops in self are finished, those in comm can start."""
        if comm.CommSize() > self.CommSize():
            raise ValueError("Cannot append a larger comm to a smaller one!")
        if len(comm.subcomms) > 0:
            raise ValueError("Cannot append a comm with subcomms, flatten first?")
        for idx, rank in enumerate(self.ranks):
            rank.Append(comm[idx])

    def Merge(self, comm):
        """Merge comm into self, such that the ops in both run in parallel."""
        if comm.CommSize() > self.CommSize():
            raise "Cannot merge a larger comm to a smaller one!"
        if len(comm.subcomms) > 0:
            raise ValueError("Cannot append a comm with subcomms, flatten first?")
        for idx, rank in enumerate(self.ranks):
            rank.Merge(comm[idx])

    def Send(self, src, dst, tag, size):
        return self[src].Send(dst, tag, size)

    def Recv(self, dst, src, tag, size):
        return self[dst].Recv(src, tag, size)

    def Calc(self, host, size):
        return self[host].Calc(size)

    def CommSize(self):
        return self.comm_size

    def CommSplit(self, color, key):
        if len(list(color)) < self.comm_size or len(list(key)) < self.comm_size:
            raise ValueError(
                "The length of color and key array must match the communicator size."
            )
        newcomms = []
        order = [
            (oldrank, color[oldrank], key[oldrank])
            for oldrank in range(0, self.comm_size)
        ]
        color_buckets = {}
        for o in order:
            if o[1] in color_buckets:
                color_buckets[o[1]].append(o)
            else:
                color_buckets[o[1]] = [o]
        for c in color_buckets.keys():
            c_list = sorted(
                color_buckets[c], key=lambda x: x[2]
            )  # sort by key within color
            nc = GoalComm(len(c_list))
            nc.base_comm = self
            for idx, r in enumerate(nc):
                r.base_rank = c_list[idx][
                    0
                ]  # store the rank the new rank had in the comm it was splitted from
            newcomms.append(nc)
        self.subcomms += newcomms
        return newcomms

    def write_goal(self, labeller=None, fh=sys.stdout, format="goal"):
        if format == "goal":
            fh.write("num_ranks " + str(len(self.ranks)) + "\n\n")
        elif format == "graphviz":
            fh.write("digraph G {\n")
        if labeller is None:
            labeller = GoalLabeller()
        for r in self.ranks:
            r.write_goal(labeller, fh, rankid=True, basecomm=self, format=format)
        if format == "graphviz":
            fh.write("}\n")
    

    def write_goal_subcomm(self, labeller, fh, rank, basecomm, format="goal"):
        """if this comm has a rank with base_rank=rank, print its goal ops without enclosing brackets"""
        for r in self.ranks:
            if r.base_rank == rank:
                r.write_goal(labeller, fh, rankid=False, basecomm=basecomm, format=format)

    def TranslateRank(self, rank, basecomm):
        """Find out the rank id of the given rank (in self) in basecomm"""
        if self == basecomm:
            return rank
        if rank == None:
            raise ValueError("Attempt to translate a non-existing rank!")
        return self.base_comm.TranslateRank(self.ranks[rank].base_rank, basecomm)


if __name__ == "__main__":
    comms = [ GoalComm(4), GoalComm(4) ]
    for c in comms:
        for i in range(1, c.CommSize()):
            c[0].Send(dst=i, tag=42, size=23)
        for i in range(1, c.CommSize()):
            c[i].Recv(src=0, tag=42, size=23)
        c.write_goal()
    comms[0].Append(comms[1])
    comms[0].write_goal()
