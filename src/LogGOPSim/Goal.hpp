#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <algorithm>

#include "Parser.hpp"

typedef Node* goalop_t;

class Goal {

	private:
		Graph graph;
		uint32_t rank;
		uint32_t num_ranks;
		const Schema* schema; // NULL writes the legacy 39-byte format

		uint32_t MaxCPU(uint32_t cpu = 0) {
			static uint32_t max_cpu = 0;
			if (cpu > max_cpu) max_cpu = cpu;
			return max_cpu;
		}
	
		uint32_t MaxNIC(uint32_t nic = 0) {
			static uint32_t max_nic = 0;
			if (nic > max_nic) max_nic = nic;
			return max_nic;
		}
			
	public:

		Goal() : rank(0), num_ranks(0), schema(NULL) {}
	
		goalop_t Send(uint32_t src, uint64_t dest, uint64_t size, uint64_t tag, uint32_t cpu, uint32_t nic) {
			
			Node* n = graph.addNode();
			
			n->Type = OPTYPE_SEND; 
			n->Peer = dest;
			n->Tag = tag;
			n->Proc = cpu;
			n->Nic = nic;
			n->Size = size;

			MaxCPU(cpu);
			MaxNIC(nic);
			
			return n;
		}

		goalop_t Recv(uint64_t src, uint32_t dest, uint64_t size, uint64_t tag, uint32_t cpu, uint32_t nic) {
		
			Node* n = graph.addNode();
			
			n->Type = OPTYPE_RECV; 
			n->Peer = src;
			n->Tag = tag;
			n->Proc = cpu;
			n->Nic = nic;
			n->Size = size;
		
			MaxCPU(cpu);
			MaxNIC(nic);	
			
			return n;
		}

		goalop_t Calc(uint32_t src, uint64_t size, uint32_t cpu, uint32_t nic) {
		
			Node* n = graph.addNode();
			
			n->Type = OPTYPE_CALC; 
			n->Peer = 0; // this optype has not real peer, i just set it so it is clearly defined
			n->Tag = 0; // this optype has not real tag, i just set it so it is clearly defined
			n->Proc = cpu;
			n->Nic = nic;
			n->Size = size;
			
			MaxCPU(cpu);
			MaxNIC(nic);

			return n;
		}

		void StartDependency(goalop_t src, goalop_t dest) {
			// a can not be executed before b is started	
			graph.addStartDependency(src, dest);
		}

		void Dependency(goalop_t src, goalop_t dest) {
			//a can not be executed before b is finished
			graph.addDependency(src, dest);
		}

		void SerializeSchedule(char* filename) {
			
			static int fd;
			
			// create/open binary schedule if it is the first rank (rank 0)
			if (rank==0) {
				fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, S_IWUSR | S_IRUSR);
				if (fd == -1) {
					fprintf(stderr, "Couldn't open %s for schedule serialization!\n", filename);		
					perror("system error message:");
					exit(EXIT_FAILURE);
				}
			}

			if (schema != NULL) {
				graph.serialize_goalx(fd, rank, num_ranks, MaxCPU(), MaxNIC(), *schema);
			} else {
				// values above 255 are rejected per node by serialize_mmap
				graph.serialize_mmap(fd, rank, num_ranks, (uint8_t) std::min(MaxCPU(), 255u), (uint8_t) std::min(MaxNIC(), 255u));
			}
			
			// close the binary schedule if it is the last rank
			if (rank == num_ranks-1) {
				close(fd);
				sync();
			}
			
		}

		void SetRank(uint32_t r) {
			rank=r;
		}

		void SetNumRanks(uint32_t nr) {
			num_ranks = nr;
		}

		void SetSchema(const Schema* s) {
			schema = s;
		}

};
