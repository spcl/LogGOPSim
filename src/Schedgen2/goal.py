import sys

class GoalOp:

    def __init__(self, rank):
        self.rank = rank
        self.label = rank.comm.GetLabel()
        self.depends_on = []

    def requires(self, required):
        self.depends_on.append(required)


class GoalSend(GoalOp):

    def __init__(self, rank, dst, tag, size):
        super().__init__(rank)
        self.dst = dst
        self.tag = tag
        self.size = size

    def write_goal(self, fh):
        fh.write("l{label}: send {size}b to {dst} tag {tag}\n".format(label=str(self.label), 
                                                                     size=str(self.size), 
                                                                     dst=str(self.dst),
                                                                     tag=str(self.tag)))

class GoalRecv(GoalOp):

    def __init__(self, rank, src, tag, size):
        super().__init__(rank)
        self.src = src
        self.tag = tag
        self.size = size

    def write_goal(self, fh):
        fh.write("l{label}: recv {size}b from {src} tag {tag}\n".format(label=str(self.label), 
                                                                       size=str(self.size), 
                                                                       src=str(self.src),
                                                                       tag=str(self.tag)))

class GoalCalc(GoalOp):

    def __init__(self, rank, size):
        super().__init__(rank)
        self.size = size

    def write_goal(self, fh):
        fh.write("l{label}: calc {size}\n".format(label=str(self.label), size=str(self.size)))



class GoalRank:

    def __init__(self, comm, rank):
        self.comm = comm
        self.rank = rank
        self.ops = []

    def Send(self, dst, tag, size):
        op = GoalSend(rank=self, dst=dst, tag=tag, size=size)
        self.ops.append(op)
        return op

    def Recv(self, src, tag, size):
        op = GoalRecv(rank=self, src=src, tag=tag, size=size)
        self.ops.append(op)
        return op

    def write_goal(self, fh):
        fh.write("rank "+str(self.rank)+" {\n")
        for op in self.ops:
            op.write_goal(fh)
        for op in self.ops:
            for req in op.depends_on:
                fh.write("l{label1} requires l{label2}\n".format(label1=op.label, label2=req.label))
        fh.write("}\n\n")


class GoalComm:

    def __init__(self, comm_size):
        self.base_comm = self
        self.coll_base_tag = 10000
        self.comm_size = comm_size
        self.subcomms = []
        self.ranks = [GoalRank(comm=self, rank=rank) for rank in range(comm_size)]
        self.next_label = 0

    def __getitem__(self, index):
        return self.ranks[index]

    def GetLabel(self):
        l = self.next_label
        self.next_label += 1
        return l

    def Send(self, src, dst, tag, size):
        return self[src].Send(dst, tag, size)

    def Recv(self, dst, src, tag, size):
        return self[dst].Recv(src, tag, size)

    def write_goal(self, fh):
        fh.write("num_ranks "+str(len(self.ranks))+"\n\n")
        for r in self.ranks:
            r.write_goal(fh)

    def CommSplit(self, color, key):
        pass



if __name__ == "__main__":
    
    comm = GoalComm(4)
    comm[0].Send(dst=1, tag=42, size=8)
    a = comm[1].Recv(src=0, tag=42, size=8)
    b = comm.Send(src=1, dst=2, tag=42, size=16)
    b.requires(a)
    comm.Recv(dst=2, src=1, tag=42, size=16)
    comm.write_goal(sys.stdout)

