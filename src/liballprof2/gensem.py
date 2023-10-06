#! /usr/bin/env python3

import re
import clang.cindex
import argparse
from collections import defaultdict
import yaml

class AllprofCodegen:

    def __init__(self, libclang_path):
        self.libclang_path=libclang_path
        self.nodes = []
        self.semantics = {}
        self.types = defaultdict(list)
        self.BLACKLISTED_FUNCTIONS = [
            'MPI_Pcontrol', #this function is not "forwardable" without more context, so we do not generate a wrapper for it
        ]

    def get_count_for_param_in_func(self, param, func):
        # TODO minimize this
        mapping = {}
        GET_NDIMS_CART_COMM = "int ndims; MPI_Cartdim_get(comm, &ndims);"
        GET_COMM_SIZE = "int size; MPI_Comm_size(comm, &size);"
        GET_NEIGH_GRAPH_COMM = "int ideg, odeg, wted; MPI_Dist_graph_neighbors_count(ideg, odeg, wted);"
        mapping[("MPI_Cart_create", "dims")] = "ndims"
        mapping[("MPI_Cart_create", "periods")] = "ndims"
        mapping[("MPI_Cart_map", "dims")] = "ndims"
        mapping[("MPI_Cart_map", "periods")] = "ndims"
        mapping[("MPI_Cart_rank", "coords")] = (GET_NDIMS_CART_COMM, "ndims")
        mapping[("MPI_Cart_sub", "remain_dims")] = (GET_NDIMS_CART_COMM, "ndims")
        mapping[("MPI_Dist_graph_create", "nodes")] = "n"
        mapping[("MPI_Dist_graph_create", "degrees")] = "n"
        mapping[("MPI_Dist_graph_create", "targets")] = "n"
        mapping[("MPI_Dist_graph_create", "weights")] = "n"
        mapping[("MPI_Dist_graph_create_adjacent", "sources")] = "indegree"
        mapping[("MPI_Dist_graph_create_adjacent", "sourceweights")] = "indegree"
        mapping[("MPI_Dist_graph_create_adjacent", "destinations")] = "outdegree"
        mapping[("MPI_Dist_graph_create_adjacent", "destweights")] = "outdegree"
        mapping[("MPI_Comm_spawn_multiple", "array_of_maxprocs")] = "count"
        mapping[("MPI_Graph_create", "index")] = "nnodes"
        mapping[("MPI_Graph_create", "edges")] = "index[nnodes-1]"
        mapping[("MPI_Graph_map", "index")] = "nnodes"
        mapping[("MPI_Graph_map", "edges")] = "index[nnodes-1]"
        mapping[("MPI_Group_excl", "ranks")] = "n"
        mapping[("MPI_Group_incl", "ranks")] = "n"
        mapping[("MPI_Group_translate_ranks", "ranks1")] = "n"
        mapping[("MPI_Type_create_darray", "gsize_array")] = "ndims"
        mapping[("MPI_Type_create_darray", "distrib_array")] = "ndims"
        mapping[("MPI_Type_create_darray", "darg_array")] = "ndims"
        mapping[("MPI_Type_create_darray", "psize_array")] = "ndims"
        mapping[("MPI_Type_create_hindexed", "array_of_blocklengths")] = "count"
        mapping[("MPI_Type_create_indexed_block", "array_of_displacements")] = "count"
        mapping[("MPI_Type_create_struct", "array_of_block_lengths")] = "count"
        mapping[("MPI_Type_create_subarray", "size_array")] = "ndims"
        mapping[("MPI_Type_create_subarray", "subsize_array")] = "ndims"
        mapping[("MPI_Type_create_subarray", "start_array")] = "ndims"
        mapping[("MPI_Type_indexed", "array_of_blocklengths")] = "count"
        mapping[("MPI_Type_indexed", "array_of_displacements")] = "count"
        mapping[("MPI_Allgatherv", "recvcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Allgatherv", "displs")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Iallgatherv", "recvcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Iallgatherv", "displs")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Alltoallv", "sendcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Alltoallv", "sdispls")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Alltoallv", "recvcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Alltoallv", "rdispls")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Ialltoallv", "sendcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Ialltoallv", "sdispls")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Ialltoallv", "recvcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Ialltoallv", "rdispls")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Alltoallw", "sendcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Alltoallw", "sdispls")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Alltoallw", "sendtypes")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Alltoallw", "recvcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Alltoallw", "rdispls")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Alltoallw", "recvtypes")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Ialltoallw", "sendcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Ialltoallw", "sdispls")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Ialltoallw", "sendtypes")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Ialltoallw", "recvcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Ialltoallw", "rdispls")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Ialltoallw", "recvtypes")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Cart_coords", "coords")] = "maxdims"
        mapping[("MPI_Cart_get", "dims")] = "maxdims"
        mapping[("MPI_Cart_get", "periods")] = "maxdims"
        mapping[("MPI_Cart_get", "coords")] = "maxdims"
        mapping[("MPI_Dist_graph_neighbors", "sources")] = "maxindegree"
        mapping[("MPI_Dist_graph_neighbors", "sourceweights")] = "maxindegree"
        mapping[("MPI_Dist_graph_neighbors", "destinations")] = "maxoutdegree"
        mapping[("MPI_Dist_graph_neighbors", "destweights")] = "maxoutdegree"
        mapping[("MPI_Comm_spawn", "argv")] = None
        mapping[("MPI_Comm_spawn", "array_of_errcodes")] = "maxprocs"
        mapping[("MPI_Comm_spawn_multiple", "array_of_commands")] = "count" # only at root
        mapping[("MPI_Comm_spawn_multiple", "array_of_argv")] = "count"     # only at root
        mapping[("MPI_Comm_spawn_multiple", "array_of_info")] = "count"     # only at root
        mapping[("MPI_Comm_spawn_multiple", "array_of_errcodes")] = "count" # only at root
        mapping[("MPI_Dims_create", "dims")] = "ndims"
        mapping[("MPI_Gatherv", "recvcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Gatherv", "displs")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Igatherv", "recvcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Igatherv", "displs")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Graph_get", "index")] = "maxindex"
        mapping[("MPI_Graph_get", "edges")] = "maxedges"
        mapping[("MPI_Graph_neighbors", "neighbors")] = "maxneighbors"
        mapping[("MPI_Group_range_excl", "ranges")] = "n"
        mapping[("MPI_Group_range_incl", "ranges")] = "n"
        mapping[("MPI_Group_translate_ranks", "ranks2")] = "n"
        mapping[("MPI_Neighbor_allgatherv", "recvcounts")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Neighbor_allgatherv", "displs")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Ineighbor_allgatherv", "recvcounts")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Ineighbor_allgatherv", "displs")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Neighbor_alltoallv", "sendcounts")] = (GET_NEIGH_GRAPH_COMM, "odeg")
        mapping[("MPI_Neighbor_alltoallv", "sdispls")] = (GET_NEIGH_GRAPH_COMM, "odeg")
        mapping[("MPI_Neighbor_alltoallv", "recvcounts")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Neighbor_alltoallv", "rdispls")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Ineighbor_alltoallv", "sendcounts")] = (GET_NEIGH_GRAPH_COMM, "odeg")
        mapping[("MPI_Ineighbor_alltoallv", "sdispls")] = (GET_NEIGH_GRAPH_COMM, "odeg")
        mapping[("MPI_Ineighbor_alltoallv", "recvcounts")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Ineighbor_alltoallv", "rdispls")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Neighbor_alltoallw", "sendcounts")] = (GET_NEIGH_GRAPH_COMM, "odeg")
        mapping[("MPI_Neighbor_alltoallw", "sdispls")] = (GET_NEIGH_GRAPH_COMM, "odeg")
        mapping[("MPI_Neighbor_alltoallw", "sendtypes")] = (GET_NEIGH_GRAPH_COMM, "odeg")
        mapping[("MPI_Neighbor_alltoallw", "recvcounts")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Neighbor_alltoallw", "rdispls")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Neighbor_alltoallw", "recvtypes")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Ineighbor_alltoallw", "sendcounts")] = (GET_NEIGH_GRAPH_COMM, "odeg")
        mapping[("MPI_Ineighbor_alltoallw", "sdispls")] = (GET_NEIGH_GRAPH_COMM, "odeg")
        mapping[("MPI_Ineighbor_alltoallw", "sendtypes")] = (GET_NEIGH_GRAPH_COMM, "odeg")
        mapping[("MPI_Ineighbor_alltoallw", "recvcounts")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Ineighbor_alltoallw", "rdispls")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Ineighbor_alltoallw", "recvtypes")] = (GET_NEIGH_GRAPH_COMM, "ideg")
        mapping[("MPI_Pack_external", "datarep")] = "strlen(datarep)"
        mapping[("MPI_Pack_external_size", "datarep")] = "strlen(datarep)"
        mapping[("MPI_Reduce_scatter", "recvcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Ireduce_scatter", "recvcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Scatterv", "sendcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Scatterv", "displs")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Iscatterv", "sendcounts")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Iscatterv", "displs")] = (GET_COMM_SIZE, "size")
        mapping[("MPI_Startall", "array_of_requests")] = "count"
        mapping[("MPI_Testall", "array_of_requests")] = "count"
        mapping[("MPI_Testall", "array_of_statuses")] = "count"
        mapping[("MPI_Testany", "array_of_requests")] = "count"
        mapping[("MPI_Testsome", "array_of_requests")] = "incount"
        mapping[("MPI_Testsome", "array_of_indices")] = "*outcount"
        mapping[("MPI_Testsome", "array_of_statuses")] = "*outcount"
        mapping[("MPI_Type_create_hindexed_block", "array_of_displacements")] = "count"
        mapping[("MPI_Type_create_hindexed", "array_of_displacements")] = "count"
        mapping[("MPI_Type_create_struct", "array_of_displacements")] = "count"
        mapping[("MPI_Type_create_struct", "array_of_types")] = "count"
        mapping[("MPI_Type_get_contents", "array_of_integers")] = "max_integers"
        mapping[("MPI_Type_get_contents", "array_of_addresses")] = "max_addresses"
        mapping[("MPI_Type_get_contents", "array_of_datatypes")] = "max_datatypes"
        mapping[("MPI_Unpack_external", "datarep")] = "strlen(datarep)"
        mapping[("MPI_Waitall", "array_of_requests")] = "count"
        mapping[("MPI_Waitany", "array_of_requests")] = "count"
        mapping[("MPI_Waitsome", "array_of_requests")] = "incount"
        mapping[("MPI_Waitsome", "array_of_indices")] = "*outcount"
        mapping[("MPI_Waitsome", "array_of_statuses")] = "*outcount"
        if (func, param) not in mapping:
            print(f"Did not find mapping[(\"{func}\", \"{param}\")] = \"\"")
            return None
        else:
            r = mapping[(func, param)]
            if type(r) is tuple:
                return r
            else:
                return (None, r)


    def generate_trace_code_for_arg(self, param_type, param_name, function_name):
        if function_name.startswith("MPI_T_"):
            return "" #ignore tool interface
        trace_code = "  {\n"
        if param_type == "int":
            trace_code += f"  WRITE_TRACE(\"%i:\", {param_name});\n"
        elif param_type == "int *":
            trace_code += f"  WRITE_TRACE(\"%i:\", *{param_name});\n"
        elif param_type == "const int[]":
            if param_name.endswith("displs") or param_name.endswith("counts"):
                trace_code += "  int comm_size;\n"
                trace_code += "  PMPI_Comm_size(comm, &comm_size);\n"
                trace_code += "  for (int i=0; i<comm_size-1; i++) {\n"
                trace_code += f"    WRITE_TRACE(\"%i,\", {param_name}[i]);\n"
                trace_code += "  }\n"
            else:
                count = self.get_count_for_param_in_func(param_name, function_name)
                trace_code += f"  for (int i=0; i<{count}; i++)"+" {\n"
                trace_code += f"    WRITE_TRACE(\"%i,\", {param_name}[i]);\n"
                trace_code += "  }\n"
        elif param_type.endswith("void *"):
            trace_code += f"  WRITE_TRACE(\"%p:\", {param_name});\n"
        elif param_type.endswith("char *"):
            trace_code += f"  WRITE_TRACE(\"%s:\", {param_name});\n"
        elif param_type == "MPI_File":
            trace_code += f"  MPI_Fint c2f_file;"
            trace_code += f"  c2f_file = PMPI_File_c2f({param_name});"
            trace_code += f"  WRITE_TRACE(\"%lli\", (long long int) c2f_file);\n"
        elif param_type == "MPI_Info":
            trace_code += f"  MPI_Fint c2f_info;"
            trace_code += f"  c2f_info = PMPI_Info_c2f({param_name});"
            trace_code += f"  WRITE_TRACE(\"%lli\", (long long int) c2f_info);\n"
        elif param_type == "MPI_Info *":
            trace_code += f"  MPI_Fint c2f_info;"
            trace_code += f"  c2f_info = PMPI_Info_c2f(*{param_name});"
            trace_code += f"  WRITE_TRACE(\"%lli\", (long long int) c2f_info);\n"
        elif param_type == "MPI_Message":
            trace_code += f"  MPI_Fint c2f_message;"
            trace_code += f"  c2f_message = PMPI_Message_c2f({param_name});"
            trace_code += f"  WRITE_TRACE(\"%lli\", (long long int) c2f_message);\n" 
        elif param_type == "MPI_Message *":
            trace_code += f"  MPI_Fint c2f_message;"
            trace_code += f"  c2f_message = PMPI_Message_c2f(*{param_name});"
            trace_code += f"  WRITE_TRACE(\"%lli\", (long long int) c2f_message);\n" 
 
        elif param_type == "MPI_Errhandler":
            trace_code += f"  WRITE_TRACE(\"%p\", (long long int) &{param_name});\n" 
        elif param_type == "MPI_Errhandler *":
            trace_code += f"  WRITE_TRACE(\"%p\", (long long int) {param_name});\n" 


        elif param_type == "MPI_Datatype":
            trace_code += f"  int ddtsize;\n"
            trace_code += f"  PMPI_Type_size({param_name}, &ddtsize);\n"
            trace_code += f"  WRITE_TRACE(\"%i:\", ddtsize);\n"
        elif param_type in "MPI_Datatype *":
            if function_name.endswith("_free"):
                trace_code += f"  WRITE_TRACE(\"%p:\", {param_name});\n"
            else:
                trace_code += f"  int ddtsize;\n"
                trace_code += f"  PMPI_Type_size(*{param_name}, &ddtsize);\n"
                trace_code += f"  WRITE_TRACE(\"%i:\", ddtsize);\n"
       
        elif param_type.endswith("_function *"):
            trace_code += f"  WRITE_TRACE(\"%p:\", {param_name});\n"
        
        elif param_type == "MPI_Win":
            # We set an info key in any function creating a new win. These keys are NOT guranteed to be globally unique, only per comm_world rank!
            trace_code += f"  int win_id=-1, val_present=0;\n"
            trace_code += f"  PMPI_Win_get_attr({param_name}, WINID_KEY, &win_id, &val_present);\n"
            trace_code += f"  WRITE_TRACE(\"%i:\", win_id);\n"
            trace_code += f"  assert(val_present);\n"
        elif param_type == "MPI_Win *":
            if function_name.endswith("_free"):
                trace_code += f"  WRITE_TRACE(\"%p:\", {param_name});\n"
            else:
                trace_code += f"  int win_id=-1, val_present=0;\n"
                trace_code += f"  PMPI_Win_get_attr(*{param_name}, COMMID_KEY, &win_id, &val_present);\n"
                trace_code += f"  WRITE_TRACE(\"%i:\", win_id);\n"
                trace_code += f"  assert(val_present);\n"

 
        elif param_type == "MPI_Comm *":
            if function_name.endswith("_free"):
                trace_code += f"  WRITE_TRACE(\"%p:\", {param_name});\n"
            else:
                trace_code += f"  int comm_id, val_present;\n"
                trace_code += f"  PMPI_Comm_set_attr(*{param_name}, COMMID_KEY, &next_commid);\n"
                trace_code += f"  next_commid += 1;\n"
                trace_code += f"  PMPI_Comm_get_attr(*{param_name}, COMMID_KEY, &comm_id, &val_present);\n"
                trace_code += f"  WRITE_TRACE(\"%i:\", comm_id);\n"
                trace_code += f"  assert(val_present);\n"

        elif param_type == "MPI_Group":
            trace_code += f"  int group_c2f, group_rank, group_size;\n"
            trace_code += f"  group_c2f = PMPI_Group_c2f({param_name});\n"
            trace_code += f"  PMPI_Group_rank({param_name}, &group_rank);\n"
            trace_code += f"  PMPI_Group_size({param_name}, &group_size);\n"
            trace_code += f"  WRITE_TRACE(\"%i,%i,%i:\", group_c2f, group_rank, group_size);\n"
        elif param_type == "MPI_Group *":
            if function_name.endswith("_free"):
                trace_code += f"  WRITE_TRACE(\"%p:\", {param_name});\n"
            else:
                trace_code += f"  int group_c2f, group_rank, group_size;\n"
                trace_code += f"  group_c2f = PMPI_Group_c2f(*{param_name});\n"
                trace_code += f"  PMPI_Group_rank(*{param_name}, &group_rank);\n"
                trace_code += f"  PMPI_Group_size(*{param_name}, &group_size);\n"
                trace_code += f"  WRITE_TRACE(\"%i,%i,%i:\", group_c2f, group_rank, group_size);\n"

        elif param_type in ["MPI_Aint", "MPI_Offset", "MPI_Op", "MPI_Count"]:
            trace_code += f"  WRITE_TRACE(\"%llu:\", (long long unsigned int) {param_name});\n"
        elif param_type in ["MPI_Aint *", "MPI_Count *", "MPI_Offset *"]:
            trace_code += f"  WRITE_TRACE(\"%llu:\", (long long unsigned int) *{param_name});\n"
        elif param_type in ["const MPI_Aint[]"]:
            trace_code += f"  WRITE_TRACE(\"%llu:\", (long long unsigned int) *{param_name});\n"
        elif param_type in ["MPI_Request *", "MPI_Request[]"]:
            trace_code += f"  WRITE_TRACE(\"%p:\", {param_name});\n"
        elif param_type in ["MPI_Status *", "MPI_Status[]", "const MPI_Status *"]:
            trace_code += f"  WRITE_TRACE(\"%p:\", {param_name});\n"
        else:
            print(f"Tracing for [{param_type}] not implemented! (appears in {function_name})")
            #print(f"Tracing for [{param_type}] not implemented!")
        trace_code += "  }\n"
        return trace_code



    def traverse_ast(self, node, depth=0, print_ast=False):
        if print_ast:
            print('  ' * depth + f'{node.kind} ({node.displayname})')
        if node.kind is clang.cindex.CursorKind.FUNCTION_DECL and re.match("MPI_.*", node.displayname) :
            self.nodes += [node]
        for child in node.get_children():
            self.traverse_ast(child, depth + 1, print_ast)


    def semnatics_for_func(self, node):
        function_name = node.spelling
        return_type = node.result_type.spelling
        if function_name.startswith("MPI_T_") or function_name == "MPI_Pcontrol":
            return
        self.semantics[function_name] = {}
        self.semantics[function_name]['return_type'] = return_type
        self.semantics[function_name]['params'] = []
        
        for param_cursor in node.get_children():
            param_dict = {}
            if param_cursor.kind != clang.cindex.CursorKind.PARM_DECL:
                continue
            param_type = param_cursor.type.spelling
            param_name = param_cursor.spelling                
            param_dict['name'] = param_name
            param_dict['type'] = param_type
            if "[]" in param_type:
                prolog, varname = self.get_count_for_param_in_func(param=param_name, func=function_name)
                param_dict['elem_count'] = varname
                param_dict['prolog_elem_count'] = prolog
                param_dict['trace_each_elem'] = True
            self.semantics[function_name]['params'].append(param_dict)


    def process_func(self, node, mode):
        if mode == 'semantics':
           self.semnatics_for_func(node)


    def process_header(self, filename, mode):
        clang.cindex.Config.set_library_path(self.libclang_path)
        index = clang.cindex.Index.create()
        translation_unit = index.parse(filename)
        if not translation_unit:
            print("Error parsing the file.")
            return
        root_cursor = translation_unit.cursor
        self.traverse_ast(root_cursor)
        for node in self.nodes:
            self.process_func(node, mode)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='liballprof_gencode',
                    description='Generates wrappers for the MPI functions present in the supplied MPI header file. The wrappers output in liballprof2 trace format.',
                    epilog='')
    parser.add_argument('-m', '--mpi-header',           default="mpi.h",                    help="MPI header file to use as input (default: mpi.h)")
    parser.add_argument('-s', '--semantics-file',       default='mpi_sem.yml',              help="Name of the file that specifies the tracer semantics (default: mpi-sem.yml)")
    parser.add_argument('-l', '--libclang-path',        default="",                         help="Path to libclang, if empty let clang python module guess. (default=\"\")")
    args = parser.parse_args()

    codegen = AllprofCodegen(libclang_path=args.libclang_path)
    codegen.outfile = open(args.semantics_file, "w")
    codegen.process_header(args.mpi_header, mode='semantics')
    codegen.outfile.write(yaml.dump(codegen.semantics))
    codegen.outfile.close()
