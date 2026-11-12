#include <map>
#include <vector>
#include <string>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/types.h>

#include "LogGOPSim.hpp"
#include "TraceSchema.hpp"

#define MAGIC_COOKIE 4223
#define MAGIC_COOKIE_INVALID 2342


#define OPTYPE_SEND 1
#define OPTYPE_RECV 2
#define OPTYPE_CALC 3

typedef uint64_t base_t;

// TODO Experiment with packing of the node structure,
//      that could save us up to 30% in space, but might
//      harm time.

struct Node {
	
	uint64_t           Size;
	std::vector<Node*> DependOnMe;
	std::vector<Node*> StartDependOnMe;
	uint32_t           DependenciesCnt;
	uint32_t           offset;
	// wider than in the binary, so that the writer can range-check instead of truncating
	uint64_t           Peer;
	uint64_t           Tag;
	uint32_t           Proc;
	uint32_t           Nic;
	char               Type;
	
};

struct DeserializedNode {

	uint32_t           DependenciesCnt;
	char               Type;
	uint32_t           Peer;
	uint64_t           Size;
	uint32_t           Tag;
	uint8_t            Proc;
	uint8_t            Nic;
	uint32_t           offset;
	std::vector<uint32_t> DependOnMe;
	std::vector<uint32_t> StartDependOnMe;
};

class Graph {
	
	private:
	
	std::vector<Node*> RootNodes;
	std::vector<Node*> allNodes;

	//uint32_t num_nodes;
	uint32_t num_edges;
	uint32_t offset_cntr;

	char* mapping_start;

/*
	uint64_t get_file_size(int fd) {
		
		struct stat f_info;
		int r = fstat(fd, &f_info);
		assert(r == 0);
		return f_info.st_size;
	}
*/

	void find_root_nodes() {

		RootNodes.clear();

		for (std::vector<Node*>::iterator it =  allNodes.begin(); it != allNodes.end(); it++) {
			if ((**it).DependenciesCnt == 0) RootNodes.push_back(*it);
		}

	}

	public:

	~Graph() {
		for (std::vector<Node*>::iterator it = allNodes.begin(); it != allNodes.end(); it++) {
			delete *it;
		}
	}

	Graph() {
	//	num_nodes = 0;
		num_edges = 0;
		offset_cntr = 0;
	}
	
	inline Node* addNode() {
		
		/**
			add a new node to the graph
		*/
		
		Node* N = new Node;
		N->DependenciesCnt = 0;
		N->offset = offset_cntr;
		offset_cntr++;
		allNodes.push_back(N);
		
		return N;
	}

	inline void addDependency(Node* a, Node* b) {
		
		/** 
		   addDependency(a,b) means that a can not be
		   executed before b is finished 
		*/

		b->DependOnMe.push_back(a);
		a->DependenciesCnt++;
		num_edges++;
	}

	inline void addStartDependency(Node* a, Node *b) {
		
		/** 
		   addStartDependency(a,b) means that a can not be
		   executed before b is started
		*/

		b->StartDependOnMe.push_back(a);
		a->DependenciesCnt++;
		num_edges++;
	}

	void write_as_dot() {
		/** 
			Produces a dot representation of the graph. This is usefull for debugging purposes.
		*/
		
		FILE* fd = fopen("graph.dot", "w");
		assert(fd != NULL);
		fprintf(fd, "digraph mygraph {\n");
		fprintf(fd, "graph [rankdir=LR];\n");
		fprintf(fd, "node [shape=record];\n");
					
			for (std::vector<Node*>::iterator it = allNodes.begin(); it != allNodes.end(); it++) {
				char typestr[5];
				if ((**it).Type == OPTYPE_SEND) strcpy(typestr, "Send");
				else if ((**it).Type == OPTYPE_RECV) strcpy(typestr, "Recv");
				else if ((**it).Type == OPTYPE_CALC) strcpy(typestr, "Calc");
				else strcpy(typestr, "Unkn");
				fprintf(fd, "%i [label=\"<f0> Type: %s | <f1> Peer: %llu | <f2> Size: %lu | <f3> Tag: %llu | <f4> Proc: %u | <f5> Nic: %u \"]\n", 
							 (**it).offset,    typestr,        (unsigned long long) (**it).Peer, (unsigned long int) (**it).Size,  (unsigned long long) (**it).Tag  , (**it).Proc,     (**it).Nic);
			}

			for (std::vector<Node*>::iterator it = allNodes.begin(); it != allNodes.end(); it++) {
				for (std::vector<Node*>::iterator dit = (**it).DependOnMe.begin(); dit != (**it).DependOnMe.end(); dit++) {
					fprintf(fd, "%i:f0 -> %i:f0\n", (**it).offset, (**dit).offset );
				}
				for (std::vector<Node*>::iterator dit = (**it).StartDependOnMe.begin(); dit != (**it).StartDependOnMe.end(); dit++) {
					fprintf(fd, "%i:f0 -> %i:f0 [arrowhead=diamond]\n", (**it).offset, (**dit).offset );
				}
			}

		fprintf(fd, "} \n");

	}

/*
	void serialize(FILE* fd, uint32_t rank, uint32_t num_ranks) {

		uint32_t buf32;

		rewind(fd);

		if (rank == 0) {
			// make room for jumptable
			fseek(fd, sizeof(uint64_t)*2*num_ranks + sizeof(uint32_t), SEEK_SET);
		}
		else {
			// jump to the end of the file
			fseek(fd, sizeof(uint32_t) + (2*(rank-1)+1)*sizeof(uint64_t), SEEK_SET);
			uint64_t eos;
			fread(&eos, sizeof(uint64_t), 1, fd);
			fseek(fd, eos, SEEK_SET);
		}
		
		find_root_nodes();
	
		long start = ftell(fd);
		// appendix starts right after all nodes
		long pos_in_appendix = start + 
		                       sizeof(uint32_t)*2 + // for Nodecount and Independent Actions count
							   sizeof(uint32_t)*RootNodes.size() + // for indp. actions offsets
							   (sizeof(char) + sizeof(uint8_t)*2 + sizeof(uint32_t)*7 + sizeof(uint64_t)) * allNodes.size(); // for actual nodeinfo
		// number of elements in the appendix so far
		uint32_t num_in_appendix = 0;

		// number of nodes in the schedule
		buf32 = (uint32_t) allNodes.size();
		fwrite( &buf32, sizeof(uint32_t), 1, fd );


		// write independant actions offsets
		buf32 = RootNodes.size(); 
		fwrite(&buf32, sizeof(uint32_t), 1, fd); // number of independent actions
		for (std::vector<Node*>::iterator it = RootNodes.begin(); it != RootNodes.end(); it++) {
			buf32 = (**it).offset;
			fwrite(&buf32, sizeof(uint32_t), 1, fd); // offset of independent action
		}
		

		Node n;
		int cnt = 0;
		for (std::vector<Node*>::iterator it = allNodes.begin(); it != allNodes.end(); it++) {
			
			cnt++;

			// write the actual operation info - this is a bit complicated thanks to padding that the compiler might do...
			fwrite( &((**it).DependenciesCnt), sizeof(uint32_t), 1, fd );
			fwrite( &((**it).Type), sizeof(char), 1, fd );
			fwrite( &((**it).Peer), sizeof(uint32_t), 1, fd );
			fwrite( &((**it).Size), sizeof(uint64_t), 1, fd );
			fwrite( &((**it).Tag), sizeof(uint32_t), 1, fd );
			fwrite( &((**it).Proc), sizeof(uint8_t), 1, fd );
			fwrite( &((**it).Nic), sizeof(uint8_t), 1, fd );
			
			// handling dependencies
			buf32 = (**it).DependOnMe.size();
			fwrite( &buf32, sizeof(uint32_t), 1, fd ); // number of ops depending on this node
			
			if ((**it).DependOnMe.size() > 0) {
				fwrite( &num_in_appendix, sizeof(uint32_t), 1, fd ); // start of dependant offsets in appendix
			} 
			else {
				uint32_t b = 999;
				fwrite( &b, sizeof(uint32_t), 1, fd ); // undefined
			}
			
			long pos = ftell(fd);
			// write offsets of depending ops into appendix
			fseek(fd, pos_in_appendix, SEEK_SET);
			for (std::vector<Node*>::iterator dit = (**it).DependOnMe.begin(); dit != (**it).DependOnMe.end(); dit++) {
				fwrite( &((**dit).offset), sizeof(uint32_t), 1, fd );
			}
			// adjust number of elements and position in appendix
			num_in_appendix += (**it).DependOnMe.size();
			pos_in_appendix = ftell(fd);
			// jump back to the node info (we're in appendix right now)
			fseek(fd, pos, SEEK_SET);
			
			// handling start-dependencies
			buf32 = (**it).StartDependOnMe.size();
			fwrite( &buf32, sizeof(uint32_t), 1, fd ); // number of ops start-depending on this node
			if ((**it).StartDependOnMe.size() > 0) { 
				fwrite( &num_in_appendix, sizeof(uint32_t), 1, fd ); // start of start-dependant offsets in appendix
			}
			else {
				uint32_t b = -1;
				fwrite( &b, sizeof(uint32_t), 1, fd ); // undefinedx
			}
			pos =  ftell(fd);
			// write offsets of start-depending ops into appendix
			fseek(fd, pos_in_appendix, SEEK_SET);
			for (std::vector<Node*>::iterator dit = (**it).StartDependOnMe.begin(); dit != (**it).StartDependOnMe.end(); dit++) {
				fwrite( &((**dit).offset), sizeof(uint32_t), 1, fd );
			}
			// adjust number of elements and position in appendix
			num_in_appendix += (**it).StartDependOnMe.size();
			pos_in_appendix = ftell(fd);
			// jump back to the node info (we're in appendix right now)
			fseek(fd, pos, SEEK_SET);
		
		}
		
		// write jumptable info: the schedule started at start and ended at pos_in_appendix
		fseek(fd, 0, SEEK_SET);
		
		// the number of ranks in this file
		buf32 = num_ranks;
		fwrite(&buf32, sizeof(uint32_t), 1, fd);
		fseek(fd, sizeof(uint64_t)*2*rank + sizeof(uint32_t), SEEK_SET);
		uint64_t buf64 = start;
		fwrite(&buf64, sizeof(uint64_t), 1, fd);
		buf64 = pos_in_appendix;
		fwrite(&buf64, sizeof(uint64_t), 1, fd);
	}

*/

	void serialize_mmap(int fd, uint32_t rank, uint32_t num_ranks, uint8_t max_cpu, uint8_t max_nic) {
		
		char *start_rankdata;
		uint64_t end_of_lastrank;
		static uint64_t filesize;
		char *pos;

		find_root_nodes();

		if (rank == 0) {
			// calculate the size of the file
			filesize = 0;
			filesize += sizeof(uint64_t); // magic cookie
			filesize += sizeof(uint32_t); // num ranks
			filesize += sizeof(uint8_t); // max_cpu
			filesize += sizeof(uint8_t); // max_nic
			filesize += sizeof(uint64_t)*2*num_ranks; // jumptable
			filesize += sizeof(uint32_t); // num nodes
			filesize += sizeof(uint32_t); // num indp actions
			filesize += (sizeof(uint32_t)*RootNodes.size()); // rootnodes offsets
			filesize += (sizeof(char)+sizeof(uint8_t)*2+sizeof(uint32_t)*7+sizeof(uint64_t))*allNodes.size(); // nodeinfo
			filesize += (sizeof(uint32_t)*num_edges); //appendix

			// enlarge the file
			lseek(fd, filesize-1, SEEK_SET);
			int r = write(fd, "", 1);
			assert(r == 1);
			
			// mmap the file
			mapping_start = (char*) mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
			if (mapping_start == MAP_FAILED) {
				perror("couldn't mmap the output file");
				exit(EXIT_FAILURE);
			}
			*( (uint64_t*) mapping_start ) = (uint64_t) MAGIC_COOKIE;
			mapping_start += sizeof(uint64_t); // jump over magic cookie
			end_of_lastrank = sizeof(uint32_t) + sizeof(uint8_t)*2 + sizeof(uint64_t)*2*num_ranks;
			start_rankdata = mapping_start + sizeof(uint32_t) + sizeof(uint8_t)*2 + sizeof(uint64_t)*2*num_ranks;  // our rankdata starts right after the jumptable
			
		}
		else {
			filesize += sizeof(uint32_t); // num nodes
			filesize += sizeof(uint32_t); // num indp actions
			filesize += sizeof(uint32_t)*RootNodes.size(); // rootnodes offsets
			filesize += (sizeof(char)+sizeof(uint8_t)*2+sizeof(uint32_t)*7+sizeof(uint64_t))*allNodes.size(); // nodeinfo
			filesize += sizeof(uint32_t)*num_edges; //appendix


			// enlarge the file
			lseek(fd, filesize-1, SEEK_SET);
			int r = write(fd, "", 1);
			assert(r == 1);

			// mmap the file
			mapping_start = (char*) mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
			if (mapping_start == MAP_FAILED) {
				perror("mmap failed");
				exit(EXIT_FAILURE);
			}

			mapping_start += sizeof(uint64_t); // jump over magic cookie

			end_of_lastrank = *((uint64_t*) (mapping_start + sizeof(uint32_t) + sizeof(uint8_t)*2 + sizeof(uint64_t)*(2*(rank-1)+1))); 
			start_rankdata = mapping_start + end_of_lastrank;
		}

		pos = start_rankdata;
		uint32_t num_in_appendix = 0;

		*((uint32_t*) pos) = (uint32_t) allNodes.size();	 	pos += sizeof(uint32_t);	// number of nodes in the schedule
		*((uint32_t*) pos) = (uint32_t) RootNodes.size();		pos += sizeof(uint32_t);	// number of independent actions
		
		// independent action offsets
		for (std::vector<Node*>::iterator it = RootNodes.begin(); it != RootNodes.end(); it++) {
			*((uint32_t*) pos) = (**it).offset; pos += sizeof(uint32_t);
		}

		// node data 

		for (std::vector<Node*>::iterator it = allNodes.begin(); it != allNodes.end(); it++) {

			// the legacy record has fixed widths, refuse values that would be truncated
			check_value("peer", (**it).Peer, 32, rank, (**it).offset);
			check_value("tag",  (**it).Tag,  32, rank, (**it).offset);
			check_value("cpu",  (**it).Proc, 8,  rank, (**it).offset);
			check_value("nic",  (**it).Nic,  8,  rank, (**it).offset);

			*((uint32_t*) pos) = (**it).DependenciesCnt; 		pos += sizeof(uint32_t);	// number of actions this action depends on
			*((char*) pos)     = (**it).Type;            		pos += sizeof(char);		// type of this action
			*((uint32_t*) pos) = (**it).Peer;            		pos += sizeof(uint32_t);	// peer of this action
			*((uint64_t*) pos) = (**it).Size;            		pos += sizeof(uint64_t);	// size of the data transfer / length of the local calculation
			*((uint32_t*) pos) = (**it).Tag;             		pos += sizeof(uint32_t);	// tag of the send/recv operation
			*((uint8_t*) pos)  = (**it).Proc;            		pos += sizeof(uint8_t);		// processor used for this action
			*((uint8_t*) pos)  = (**it).Nic;             		pos += sizeof(uint8_t);		// network interface used for this action
			*((uint32_t*) pos) = (**it).DependOnMe.size();		pos += sizeof(uint32_t);	// number of actions that depend on this actions termination
			*((uint32_t*) pos) = num_in_appendix;				pos += sizeof(uint32_t);	// start index of dependent actions (in appendix)
			num_in_appendix += (**it).DependOnMe.size();
			*((uint32_t*) pos) = (**it).StartDependOnMe.size();	pos += sizeof(uint32_t);	// number of actions that depend on this actions start
			*((uint32_t*) pos) = num_in_appendix;				pos += sizeof(uint32_t);	// start index of start-dependent actions (in appendix)
			num_in_appendix += (**it).StartDependOnMe.size();
		}
		
		// appendix data
	
		for (std::vector<Node*>::iterator it = allNodes.begin(); it != allNodes.end(); it++) {
			for (std::vector<Node*>::iterator dit = (**it).DependOnMe.begin(); dit != (**it).DependOnMe.end(); dit++) {
				*((uint32_t*) pos) = (**dit).offset;       		pos += sizeof(uint32_t);	// offset of dependent action
			}
			for (std::vector<Node*>::iterator dit = (**it).StartDependOnMe.begin(); dit != (**it).StartDependOnMe.end(); dit++) {
				*((uint32_t*) pos) = (**dit).offset;       		pos += sizeof(uint32_t);	// offset of start-dependent action
			}
		}

		// jumptable info
		*((uint32_t*) mapping_start) = num_ranks;														// number of ranks in this schedule-file

		*((uint8_t*) (mapping_start + sizeof(uint32_t) )) = max_cpu;	// minimal number of cpu required to simulate this schedule
		*((uint8_t*) (mapping_start + sizeof(uint32_t) + sizeof(uint8_t) )) = max_nic;	// minimal number of nics required to simulate this schedule
		
		*((uint64_t*) (mapping_start + sizeof(uint32_t) + sizeof(uint8_t)*2 + sizeof(uint64_t)*2*rank)) = end_of_lastrank;	// start of this ranks info
		*((uint64_t*) (mapping_start + sizeof(uint32_t) + sizeof(uint8_t)*2 + sizeof(uint64_t)*(2*rank+1))) = pos - mapping_start;			// end of this ranks info
	
		//printf("s: %llu e: %llu\n", (long long unsigned int) end_of_lastrank, (long long unsigned int) (pos - mapping_start));

		// munmap the files so that the contents get written
		int r = munmap(mapping_start-sizeof(uint64_t), (size_t) (pos - mapping_start));
		assert(r == 0);
	}

	static void check_value(const char* field, uint64_t value, uint8_t width, uint32_t rank, uint32_t node) {
		if (!check_fits(value, width)) {
			fprintf(stderr, "Error: field '%s' value %llu does not fit in %u bits (rank %u, node %u)\n",
			        field, (unsigned long long) value, width, rank, node);
			exit(EXIT_FAILURE);
		}
	}

	/* encodes one value into a GOALX record, refusing anything the declared width cannot hold */
	static void encode_field(char* rec, const FieldDesc* f, const char* name, uint64_t value, uint32_t rank, uint32_t node) {
		if (f == NULL) {
			// an omitted field decodes as 0, so only 0 may be dropped
			if (value != 0) {
				fprintf(stderr, "Error: field '%s' is not in the schema, but rank %u node %u uses value %llu\n",
				        name, rank, node, (unsigned long long) value);
				exit(EXIT_FAILURE);
			}
			return;
		}
		uint64_t mask = width_mask(f->width);
		if ((f->flags & FF_ANY_SENTINEL) && value == 0xFFFFFFFFULL) {
			value = mask;
		} else if (value > mask || ((f->flags & FF_ANY_SENTINEL) && value == mask)) {
			fprintf(stderr, "Error: field '%s' value %llu does not fit in %u bits%s (rank %u, node %u)\n",
			        name, (unsigned long long) value, f->width,
			        value == mask ? ", all ones is reserved for ANY" : "", rank, node);
			exit(EXIT_FAILURE);
		}
		put_bits(rec, f->bit_offset, f->width, value);
	}

	void serialize_goalx(int fd, uint32_t rank, uint32_t num_ranks, uint32_t max_cpu, uint32_t max_nic, const Schema& schema) {

		static uint64_t filesize;
		static uint32_t header_len;
		std::vector<char> header;
		uint64_t start;

		find_root_nodes();

		if (rank == 0) {
			schema.serialize_header(header, num_ranks, max_cpu, max_nic);
			header_len = header.size();
			start = header_len + sizeof(uint64_t)*2*num_ranks; // rank data starts right after the jumptable
		} else {
			start = filesize;
		}

		const uint64_t rb = schema.record_bytes;
		filesize = start;
		filesize += sizeof(uint32_t)*2; // num nodes, num roots
		filesize += sizeof(uint32_t)*RootNodes.size(); // root node offsets
		filesize += sizeof(uint32_t)*allNodes.size(); // initial dependency counters
		filesize += rb*allNodes.size(); // records
		filesize += sizeof(uint32_t)*num_edges; // appendix
		filesize += GOALX_PADDING;

		if (ftruncate(fd, filesize) != 0) {
			perror("couldn't enlarge the output file");
			exit(EXIT_FAILURE);
		}
		char* map = (char*) mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (map == MAP_FAILED) {
			perror("couldn't mmap the output file");
			exit(EXIT_FAILURE);
		}
		if (rank == 0) memcpy(map, header.data(), header.size());

		char* pos = map + start;
		uint32_t buf32;

		buf32 = allNodes.size();	memcpy(pos, &buf32, sizeof(uint32_t));	pos += sizeof(uint32_t);
		buf32 = RootNodes.size();	memcpy(pos, &buf32, sizeof(uint32_t));	pos += sizeof(uint32_t);
		for (std::vector<Node*>::iterator it = RootNodes.begin(); it != RootNodes.end(); it++) {
			memcpy(pos, &(**it).offset, sizeof(uint32_t));	pos += sizeof(uint32_t);
		}
		for (std::vector<Node*>::iterator it = allNodes.begin(); it != allNodes.end(); it++) {
			memcpy(pos, &(**it).DependenciesCnt, sizeof(uint32_t));	pos += sizeof(uint32_t);
		}

		const FieldDesc* f[FK_NUM_BUILTIN];
		for (int k = 0; k < FK_NUM_BUILTIN; k++) f[k] = schema.builtin((FieldKind) k);

		std::vector<char> rec(rb + GOALX_PADDING + 1);
		uint32_t num_in_appendix = 0;
		for (std::vector<Node*>::iterator it = allNodes.begin(); it != allNodes.end(); it++) {
			Node& n = **it;
			uint32_t ndep = n.DependOnMe.size();
			uint32_t nsdep = n.StartDependOnMe.size();

			std::fill(rec.begin(), rec.end(), 0);
			encode_field(&rec[0], f[FK_TYPE],  "type",  (uint8_t) n.Type, rank, n.offset);
			encode_field(&rec[0], f[FK_PEER],  "peer",  n.Peer,           rank, n.offset);
			encode_field(&rec[0], f[FK_SIZE],  "size",  n.Size,           rank, n.offset);
			encode_field(&rec[0], f[FK_TAG],   "tag",   n.Tag,            rank, n.offset);
			encode_field(&rec[0], f[FK_CPU],   "cpu",   n.Proc,           rank, n.offset);
			encode_field(&rec[0], f[FK_NIC],   "nic",   n.Nic,            rank, n.offset);
			encode_field(&rec[0], f[FK_NDEP],  "ndep",  ndep,             rank, n.offset);
			encode_field(&rec[0], f[FK_DEPIX], "depix", num_in_appendix, rank, n.offset);
			encode_field(&rec[0], f[FK_NSDEP], "nsdep", nsdep,            rank, n.offset);
			if (f[FK_SDEPIX] != NULL) encode_field(&rec[0], f[FK_SDEPIX], "sdepix", num_in_appendix + ndep, rank, n.offset);
			for (size_t i = 0; i < schema.fields.size(); i++) {
				const FieldDesc& c = schema.fields[i];
				if (c.kind == FK_CUSTOM) encode_field(&rec[0], &c, c.name.c_str(), c.dflt, rank, n.offset);
			}
			memcpy(pos, &rec[0], rb);	pos += rb;
			num_in_appendix += ndep + nsdep;
		}

		for (std::vector<Node*>::iterator it = allNodes.begin(); it != allNodes.end(); it++) {
			for (std::vector<Node*>::iterator dit = (**it).DependOnMe.begin(); dit != (**it).DependOnMe.end(); dit++) {
				memcpy(pos, &(**dit).offset, sizeof(uint32_t));	pos += sizeof(uint32_t);
			}
			for (std::vector<Node*>::iterator dit = (**it).StartDependOnMe.begin(); dit != (**it).StartDependOnMe.end(); dit++) {
				memcpy(pos, &(**dit).offset, sizeof(uint32_t));	pos += sizeof(uint32_t);
			}
		}
		memset(pos, 0, GOALX_PADDING);	pos += GOALX_PADDING;
		assert((uint64_t) (pos - map) == filesize);

		// the max cpu/nic grow while ranks are appended, so they are rewritten every time
		memcpy(map + GOALX_OFF_MAX_CPU, &max_cpu, sizeof(uint32_t));
		memcpy(map + GOALX_OFF_MAX_NIC, &max_nic, sizeof(uint32_t));

		uint64_t jump[2] = { start, filesize };
		memcpy(map + header_len + sizeof(uint64_t)*2*rank, jump, sizeof(jump));

		int r = munmap(map, filesize);
		assert(r == 0);
	}

};

/* decoding information shared by all ranks of one trace, computed once from the schema */
enum DecoderKind {
	DECODER_INTERP, // schema-agnostic: precomputed shift and mask per field, any layout
	DECODER_FIXED   // the original hand-written decoder, constant offsets, legacy files only
};

struct NodeLayout {

	Accessor type, peer, size, tag, cpu, nic, ndep, depix, nsdep, sdepix;
	uint32_t record_bytes;
	bool     legacy; // depcnt lives at byte 0 of each record, inside the mapping
	DecoderKind decoder;

	NodeLayout() : record_bytes(0), legacy(true), decoder(DECODER_INTERP) {}

	NodeLayout(const Schema& s, DecoderKind d) :
		type(s.builtin(FK_TYPE)), peer(s.builtin(FK_PEER)), size(s.builtin(FK_SIZE)),
		tag(s.builtin(FK_TAG)), cpu(s.builtin(FK_CPU)), nic(s.builtin(FK_NIC)),
		ndep(s.builtin(FK_NDEP)), depix(s.builtin(FK_DEPIX)), nsdep(s.builtin(FK_NSDEP)),
		sdepix(s.builtin(FK_SDEPIX)), record_bytes(s.record_bytes), legacy(s.legacy), decoder(d) {}
};

class SerializedGraph {
	
	private:
	
	const NodeLayout* layout;
	const char* roots;
	char* records;
	const char* appendix;
	uint32_t num_root_nodes;
	uint32_t num_nodes;
	uint32_t my_rank;

	// GOALX only: the mutable dependency counters, so the trace itself stays read-only
	std::vector<uint32_t> depcnt;

	std::vector<DeserializedNode> executableNodes;

	static inline uint32_t load_u32(const char* p) {
		uint32_t v;
		memcpy(&v, p, sizeof(v));
		return v;
	}

	inline uint32_t* depcnt_ptr(uint32_t offset) {
		if (layout->legacy) return (uint32_t*) (records + (size_t) layout->record_bytes * offset);
		return &depcnt[offset];
	}

	void add_root_nodes() {
	
		for (uint32_t cnt=0; cnt<num_root_nodes; cnt++) {
			DeserializedNode N = get_node_by_offset(load_u32(roots + cnt*sizeof(uint32_t)));
			executableNodes.push_back(N);
		}

	}

	/* the schema-agnostic interpreter: one precomputed accessor per field the simulator uses */
	/* decodes the payload and edge-list fields of one record; the branch is constant for a run */
	inline void decode_record(const char* rec, DeserializedNode& N, uint32_t& num_deps, uint32_t& deps_ix,
	                          uint32_t& num_startdeps, uint32_t& startdeps_ix) const {

		if (layout->decoder == DECODER_FIXED) {
			// the original decoder for the 39-byte legacy record
			N.Type = (char) *(rec + sizeof(uint32_t)); // after depcnt
			N.Peer = (uint32_t) *( (uint32_t*) (rec + sizeof(uint32_t) + sizeof(char)) ); // after depcnt + type
			N.Size = (uint64_t) *( (uint64_t*) (rec + sizeof(uint32_t) + sizeof(char) + sizeof(uint32_t)));
			N.Tag = (uint32_t) *( (uint32_t*) (rec + sizeof(uint32_t) + sizeof(char) + sizeof(uint32_t) + sizeof(uint64_t)));
			N.Proc = (uint8_t) *( (uint8_t*) (rec + sizeof(uint32_t) + sizeof(char) + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t)));
			N.Nic = (uint8_t) *( (uint8_t*) (rec + sizeof(uint32_t) + sizeof(char) + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint8_t)));
			num_deps =      (uint32_t) *( (uint32_t*) (rec + sizeof(char) + sizeof(uint64_t) + sizeof(uint32_t)*3 + sizeof(uint8_t)*2));
			deps_ix =       (uint32_t) *( (uint32_t*) (rec + sizeof(char) + sizeof(uint64_t) + sizeof(uint32_t)*4 + sizeof(uint8_t)*2));
			num_startdeps = (uint32_t) *( (uint32_t*) (rec + sizeof(char) + sizeof(uint64_t) + sizeof(uint32_t)*5 + sizeof(uint8_t)*2));
			startdeps_ix =  (uint32_t) *( (uint32_t*) (rec + sizeof(char) + sizeof(uint64_t) + sizeof(uint32_t)*6 + sizeof(uint8_t)*2));
			return;
		}

		N.Type = (char) layout->type.get(rec);
		N.Peer = (uint32_t) layout->peer.get(rec);
		N.Size = layout->size.get(rec);
		N.Tag = (uint32_t) layout->tag.get(rec);
		N.Proc = (uint8_t) layout->cpu.get(rec);
		N.Nic = (uint8_t) layout->nic.get(rec);
		num_deps = (uint32_t) layout->ndep.get(rec);
		deps_ix = (uint32_t) layout->depix.get(rec);
		num_startdeps = (uint32_t) layout->nsdep.get(rec);
		startdeps_ix = layout->sdepix.present ? (uint32_t) layout->sdepix.get(rec) : deps_ix + num_deps;
	}

	DeserializedNode get_node_by_offset(uint32_t offset) {
	
		if (offset >= num_nodes) {
			fprintf(stderr, "[rank %i] got offset %i, have %i nodes\n", my_rank, offset, num_nodes);
			exit(EXIT_FAILURE);
		}

		const char* rec = records + (size_t) layout->record_bytes * offset;
		DeserializedNode N;
		uint32_t num_deps, deps_startoffset_in_apdx, num_startdeps, startdeps_startoffset_in_apdx;

		N.DependenciesCnt = *depcnt_ptr(offset);
		N.offset = offset;
		decode_record(rec, N, num_deps, deps_startoffset_in_apdx, num_startdeps, startdeps_startoffset_in_apdx);

		for (uint32_t cnt=0; cnt<num_deps; cnt++) {
			uint32_t depnode = load_u32(appendix + (size_t) (deps_startoffset_in_apdx + cnt)*sizeof(uint32_t));
			assert(depnode < num_nodes);
			N.DependOnMe.push_back(depnode);
		}
		for (uint32_t cnt=0; cnt<num_startdeps; cnt++) {
			N.StartDependOnMe.push_back(load_u32(appendix + (size_t) (startdeps_startoffset_in_apdx + cnt)*sizeof(uint32_t)));
		}
			
		return N;
	}

	/* decrements the counters of the given dependents, nodes that reach 0 become executable */
	void release(const std::vector<uint32_t>& dependents) {
		for (uint32_t cnt=0; cnt<dependents.size(); cnt++) {
			uint32_t offset = dependents[cnt];
			assert(offset < num_nodes);
			uint32_t* dep_cnt = depcnt_ptr(offset);
			(*dep_cnt)--;
			if ((*dep_cnt) == 0) {
				executableNodes.push_back(get_node_by_offset(offset));
			}
		}
	}

	public:

	uint32_t GetNumNodes() {
		return this->num_nodes;		
	}

	/* decodes every record of this rank, for measuring decoder throughput; the checksum compares decoders */
	uint64_t DecodeAll() const {
		uint64_t sum = 0;
		DeserializedNode N;
		uint32_t ndep, depix, nsdep, sdepix;
		for (uint32_t i = 0; i < num_nodes; i++) {
			decode_record(records + (size_t) layout->record_bytes * i, N, ndep, depix, nsdep, sdepix);
			sum += (uint64_t) (uint8_t) N.Type + N.Peer + N.Size + N.Tag + N.Proc + N.Nic + ndep + depix + nsdep + sdepix;
		}
		return sum;
	}

	void write_as_dot(char *filename) {
		/** 
			Produces a dot representation of the graph. This is usefull for debugging purposes.
		*/
		
		std::vector<DeserializedNode> allNodes;
		std::vector<DeserializedNode> executableNodes;

		executableNodes = GetExecutableNodes_DSN();
		while (executableNodes.size() > 0) {
			for (uint32_t cnt=0; cnt < executableNodes.size(); cnt++) {
				allNodes.push_back(executableNodes[cnt]);
				MarkNodeAsStarted_DSN(executableNodes[cnt]);
				MarkNodeAsDone_DSN(executableNodes[cnt]);
			}
			executableNodes.clear();
			executableNodes = GetExecutableNodes_DSN();
		}

		FILE* fd = fopen(filename, "w");
		assert(fd != NULL);
		fprintf(fd, "digraph mygraph {\n");
		fprintf(fd, "graph [rankdir=LR];\n");
		fprintf(fd, "node [shape=record];\n");
					
			for (std::vector<DeserializedNode>::iterator it = allNodes.begin(); it != allNodes.end(); it++) {
				char typestr[5];
				if ((*it).Type == OPTYPE_SEND) strcpy(typestr, "Send");
				else if ((*it).Type == OPTYPE_RECV) strcpy(typestr, "Recv");
				else if ((*it).Type == OPTYPE_CALC) strcpy(typestr, "Calc");
				else strcpy(typestr, "Unkn");
				fprintf(fd, "%i [label=\"<f0> Type: %s | <f1> Peer: %i | <f2> Size: %llu | <f3> Tag: %i | <f4> Proc: %i | <f5> Nic: %i \"]\n", 
							 (*it).offset,    typestr,        (*it).Peer, (unsigned long long) (*it).Size,  (*it).Tag  , (*it).Proc,     (*it).Nic);
			}

			for (std::vector<DeserializedNode>::iterator it = allNodes.begin(); it != allNodes.end(); it++) {
				for (std::vector<uint32_t>::iterator dit = (*it).DependOnMe.begin(); dit != (*it).DependOnMe.end(); dit++) {
					fprintf(fd, "%i:f0 -> %i:f0\n", (*it).offset, (*dit));
				}
				for (std::vector<uint32_t>::iterator dit = (*it).StartDependOnMe.begin(); dit != (*it).StartDependOnMe.end(); dit++) {
					fprintf(fd, "%i:f0 -> %i:f0 [arrowhead=diamond]\n", (*it).offset, (*dit));
				}
			}

		fprintf(fd, "} \n");

	}

	/* block points at this rank's block (num_nodes), map_end is the end of the mapping */
	SerializedGraph(char* block, const char* map_end, const NodeLayout* node_layout, uint32_t rank) {

		layout = node_layout;
		my_rank = rank;

		num_nodes = load_u32(block);
		num_root_nodes = load_u32(block + sizeof(uint32_t));
		roots = block + sizeof(uint32_t)*2;

		const char* depcnt_init = roots + (size_t) sizeof(uint32_t)*num_root_nodes;
		if (layout->legacy) records = (char*) depcnt_init;
		else records = (char*) depcnt_init + (size_t) sizeof(uint32_t)*num_nodes;
		appendix = records + (size_t) layout->record_bytes*num_nodes;

		if (appendix > map_end) {
			fprintf(stderr, "Error: rank %u claims %u nodes, but the trace ends before them\n", rank, num_nodes);
			exit(EXIT_FAILURE);
		}
		if (!layout->legacy) {
			depcnt.resize(num_nodes);
			if (num_nodes > 0) memcpy(&depcnt[0], depcnt_init, sizeof(uint32_t)*num_nodes);
		}

		executableNodes.clear();
		add_root_nodes();
	}

	void MarkNodeAsStarted_DSN(DeserializedNode node) {
		release(get_node_by_offset(node.offset).StartDependOnMe);
	}
		
	void MarkNodeAsDone_DSN(DeserializedNode node) {
		release(get_node_by_offset(node.offset).DependOnMe);
	}
	
	std::vector<DeserializedNode> GetExecutableNodes_DSN() { 
				
		std::vector<DeserializedNode> ret;

		for (uint32_t cnt=0; cnt<executableNodes.size(); cnt++) {	
			ret.push_back(executableNodes[cnt]);
		}
		executableNodes.clear();
		return ret;
	}

  typedef std::vector<graph_node_properties> nodelist_t;
	void GetExecutableNodes(nodelist_t *ret_ptr) {
		nodelist_t& ret = *ret_ptr;

		for (uint32_t cnt=0; cnt<executableNodes.size(); cnt++) {	
			graph_node_properties gp;
			gp.target = executableNodes[cnt].Peer;
			gp.size = executableNodes[cnt].Size;
			gp.tag = executableNodes[cnt].Tag;
			gp.proc = executableNodes[cnt].Proc;
			gp.nic = executableNodes[cnt].Nic;
			if (executableNodes[cnt].Type == OPTYPE_SEND) gp.type = OP_SEND;
			else if (executableNodes[cnt].Type == OPTYPE_RECV) gp.type = OP_RECV;
			else if (executableNodes[cnt].Type == OPTYPE_CALC) gp.type = OP_LOCOP;
			gp.offset = executableNodes[cnt].offset;
			ret.push_back(gp);
		}
		executableNodes.clear();
	}

	void MarkNodeAsStarted(uint32_t offset) {
		release(get_node_by_offset(offset).StartDependOnMe);
	}
		
	void MarkNodeAsDone(uint32_t offset) {
		release(get_node_by_offset(offset).DependOnMe);
	}

};

class Parser {

	private:

	char* mapping_start; // argh. this should be void* but c++ is anal on void*-arithmetic
	size_t mapping_length;
	uint32_t num_ranks_in_schedule;
	uint32_t max_cpu;
	uint32_t max_nic;
	FILE *schedules_fd;
	Schema schema;
	NodeLayout layout;

	uint64_t get_file_size(FILE* fd) {
		
		struct stat f_info;
		int r = fstat(fileno(fd), &f_info);
		assert(r == 0);
		return f_info.st_size;
	}

	static void fail(const char* filename, const std::string& msg) {
		fprintf(stderr, "Error: trace '%s': %s\n", filename, msg.c_str());
		exit(EXIT_FAILURE);
	}

	void open_legacy(bool save_mem) {

		uint8_t max_cpu8, max_nic8;

		fread(&num_ranks_in_schedule, sizeof(uint32_t), 1, schedules_fd);
		fread(&max_cpu8, sizeof(uint8_t), 1, schedules_fd);
		fread(&max_nic8, sizeof(uint8_t), 1, schedules_fd);
		max_cpu = max_cpu8;
		max_nic = max_nic8;

		if (save_mem == true) {
			// mmap can fail with map_private and prot_write on machines where the virtual mem is smaller than
			// the mapped region - so we fall back to map_shared. This destroys the schedule, so we invalidate
			// the magic cookie if we do this
			mapping_start = (char*) mmap(NULL, mapping_length, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(schedules_fd), 0); 
			if (mapping_start != MAP_FAILED) *((uint64_t*) mapping_start) = MAGIC_COOKIE_INVALID;
			printf("The schedule will be invalid after this simulation!\n");
		}
		
		else if (save_mem == false) {
			mapping_start = (char*) mmap(NULL, mapping_length, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(schedules_fd), 0);
			// THIS NEEDS MORE MEMORY - but it is also more convinient for interacrive use
			// because it preserves the schedules
			// Note that there is no fall-through to MAP_SHARED, we put the user in charge now!
		}
		
		if (mapping_start == MAP_FAILED) {
			fprintf(stderr, "mmap does not work on your system! Try to use the --save-mem option.\n");
			exit(EXIT_FAILURE);
		}

		schema = Schema::legacy_layout();
	}

	void open_goalx(const char* filename, bool save_mem) {

		std::string err;
		uint32_t header_len;

		if (save_mem) printf("--save-mem has no effect on GOALX traces, they are mapped read-only.\n");

		// the trace is never written: dependency counters live in a separate array
		mapping_start = (char*) mmap(NULL, mapping_length, PROT_READ, MAP_SHARED, fileno(schedules_fd), 0);
		if (mapping_start == MAP_FAILED) {
			perror("couldn't mmap the trace");
			exit(EXIT_FAILURE);
		}
		if (!Schema::parse_header(mapping_start, mapping_length, &schema, &num_ranks_in_schedule,
		                          &max_cpu, &max_nic, &header_len, &err)) {
			fail(filename, err);
		}
		if (header_len + sizeof(uint64_t)*2*(uint64_t) num_ranks_in_schedule > mapping_length) {
			fail(filename, "jump table extends past the end of the file");
		}
	}

	/* returns the start of a rank block, checked against the file size */
	char* rank_block(const char* filename, uint32_t rank) {
		uint64_t jump[2];
		if (schema.legacy) {
			// legacy jump-table offsets are relative to the byte after the magic cookie
			char* base = mapping_start + sizeof(uint64_t);
			memcpy(jump, base + sizeof(uint32_t) + sizeof(uint8_t)*2 + sizeof(uint64_t)*2*rank, sizeof(jump));
			return base + jump[0];
		}
		uint32_t header_len;
		memcpy(&header_len, mapping_start + 12, sizeof(header_len));
		memcpy(jump, mapping_start + header_len + sizeof(uint64_t)*2*rank, sizeof(jump));
		if (jump[0] > jump[1] || jump[1] > mapping_length || jump[1] - jump[0] < sizeof(uint32_t)*2) {
			fail(filename, "jump table entry of rank " + std::to_string(rank) + " is out of range");
		}
		return mapping_start + jump[0];
	}

	public:
	
  typedef std::vector<SerializedGraph> schedules_t ;
	schedules_t schedules;

	/* require_spec lists extra fields the simulation needs, e.g. "gpu_id:6" */
	/* decoder_name is "auto" (fixed for legacy files, interp for GOALX), "interp" or "fixed" */
	Parser(char* filename, bool save_mem, const char* require_spec = NULL, const char* decoder_name = "auto") {
		
		schedules_fd = fopen(filename, save_mem ? "r+" : "r");
	    
		if (schedules_fd == NULL) {
			fprintf(stderr, "Couldn't open input file %s!\n", filename);
			exit(EXIT_FAILURE);
		}

		uint64_t magic_cookie = 0;

		mapping_length = get_file_size(schedules_fd);
		fread(&magic_cookie, sizeof(uint64_t), 1, schedules_fd);
	
		if (magic_cookie == MAGIC_COOKIE_INVALID) {
			fprintf(stderr, "This is serialized goal schedule was invalidated by a prior simulation run!\n");
			exit(EXIT_FAILURE);
		}	
		if (magic_cookie == GOALX_MAGIC) {
			open_goalx(filename, save_mem);
		} else if (magic_cookie == MAGIC_COOKIE) {
			open_legacy(save_mem);
		} else {
			fprintf(stderr, "This is not a serialized goal schedule - the magic cookie is missing\n");
			exit(EXIT_FAILURE);
		}

		std::string err;
		std::vector<Requirement> extra;
		if (!parse_requirements(require_spec, &extra, &err)) fail(filename, err);
		if (!validate_schema(schema, extra, num_ranks_in_schedule, max_cpu, max_nic, &err)) fail(filename, err);

		DecoderKind decoder = schema.legacy ? DECODER_FIXED : DECODER_INTERP;
		if (decoder_name != NULL && strcmp(decoder_name, "interp") == 0) decoder = DECODER_INTERP;
		if (decoder_name != NULL && strcmp(decoder_name, "fixed") == 0) {
			if (!schema.legacy) fail(filename, "the fixed decoder only reads legacy (4223) traces");
			decoder = DECODER_FIXED;
		}

		layout = NodeLayout(schema, decoder);
		schedules.reserve(num_ranks_in_schedule);
		for (uint32_t cnt=0; cnt<num_ranks_in_schedule; cnt++) {
			schedules.emplace_back(rank_block(filename, cnt), mapping_start + mapping_length, &layout, cnt);
		}
	}

	int GetNumCPU() {
		return max_cpu+1;
	}

	int GetNumNIC() {
		return max_nic+1;
	}

	const char* GetDecoderName() {
		return layout.decoder == DECODER_FIXED ? "fixed" : "interp";
	}

	void PrintSchema(FILE* fd) {
		schema.print(fd, num_ranks_in_schedule, max_cpu, max_nic);
	}

	~Parser() {
		int r = munmap(mapping_start, mapping_length);
		assert(r == 0);
		fclose(schedules_fd);
	}
	
};
