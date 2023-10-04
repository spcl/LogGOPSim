import re
import clang.cindex
from collections import defaultdict

class AllprofCodegen:

    def __init__(self):
        self.nodes = []
        self.types = defaultdict(list)
        self.BLACKLISTED_FUNCTIONS = [
            'MPI_Pcontrol', #this function is not "forwardable" without more context, so we do not generate a wrapper for it
        ]

    def generate_pmpi_call(self, cursor):
        function_name = cursor.spelling
        return_type = cursor.result_type.spelling
        
        parameters = []
        for param_cursor in cursor.get_children():
            if param_cursor.kind == clang.cindex.CursorKind.PARM_DECL:
                param_name = param_cursor.spelling
                parameters.append(f"{param_name}")
        pmpi_code = f"  {return_type} retval = P{function_name}({', '.join(parameters)});\n"
        return pmpi_code
  
    
    def generate_trace_code_prolog(self, cursor):
        function_name = cursor.spelling
        prolog_code  = f"  WRITE_TRACE(\"%s\", \"{function_name}\");\n"
        prolog_code += f"  WRITE_TRACE(\"%0.2f:\", PMPI_Wtime()*1e6);\n"
        return prolog_code

    def generate_trace_code_epilog(self, cursor):
        function_name = cursor.spelling
        epilog_code = ""
        for param_cursor in cursor.get_children():
            if param_cursor.kind == clang.cindex.CursorKind.PARM_DECL:
                param_type = param_cursor.type.spelling
                param_name = param_cursor.spelling
                epilog_code += self.generate_trace_code_for_arg(param_type, param_name, function_name)
        epilog_code += f"  WRITE_TRACE(\"%0.2f\\n\", PMPI_Wtime()*1e6);\n"
        return epilog_code

    def generate_trace_code_for_arg(self, param_type, param_name, function_name):
        trace_code = "  {\n"
        if param_type == "int":
            trace_code += f"  WRITE_TRACE(\"%i:\", {param_name});\n"
        elif param_type == "int *":
            trace_code += f"  WRITE_TRACE(\"%i:\", *{param_name});\n"
        elif param_type == "const int[]":
            if param_name.endswith("displs") or param_name.endswith("counts"):
                trace_code += "for (int i=0; i<comm_size-1; i++) {\n"
                trace_code += f"  WRITE_TRACE(\"%i,\", {param_name}[i])\n"
                trace_code += "}\n"
            else:
                trace_code += "FIXME"
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
        elif param_type == "MPI_Message":
            trace_code += f"  MPI_Fint c2f_message;"
            trace_code += f"  c2f_message = PMPI_Message_c2f({param_name});"
            trace_code += f"  WRITE_TRACE(\"%lli\", (long long int) c2f_message);\n" 
        elif param_type == "MPI_Datatype":
            trace_code += f"  int ddtsize;\n"
            trace_code += f"  PMPI_Type_size({param_name}, &ddtsize);\n"
            trace_code += f"  WRITE_TRACE(\"%i:\", ddtsize);\n"
        elif param_type == "MPI_Datatype *":
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
                trace_code += f"  PMPI_Win_get_attr({param_name}, COMMID_KEY, &win_id, &val_present);\n"
                trace_code += f"  WRITE_TRACE(\"%i:\", win_id);\n"
                trace_code += f"  assert(val_present);\n"

        elif param_type == "MPI_Comm":
            # We set an info key in any function creating a new comm. These keys are NOT guranteed to be globally unique, only per comm_world rank!
            trace_code += f"  int comm_id, val_present, comm_rank, comm_size;\n"
            trace_code += f"  PMPI_Comm_get_attr({param_name}, COMMID_KEY, &comm_id, &val_present);\n"
            trace_code += f"  assert(val_present);\n"
            trace_code += f"  PMPI_Comm_rank({param_name}, &comm_rank);\n"
            trace_code += f"  PMPI_Comm_size({param_name}, &comm_size);\n"
            trace_code += f"  WRITE_TRACE(\"%i,%i,%i:\", comm_id, comm_rank, comm_size);\n"
        elif param_type == "MPI_Group":
            trace_code += f"  int group_c2f, comm_rank, comm_size;\n"
            trace_code += f"  group_c2f = PMPI_Group_c2f({param_name});\n"
            trace_code += f"  PMPI_Group_rank({param_name}, &group_rank);\n"
            trace_code += f"  PMPI_Group_size({param_name}, &group_size);\n"
            trace_code += f"  WRITE_TRACE(\"%i,%i,%i:\", group_c2f, group_rank, group_size);\n"
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
        elif param_type in ["MPI_Aint", "MPI_Offset", "MPI_Op", "MPI_Count"]:
            trace_code += f"  WRITE_TRACE(\"%llu:\", (long long unsigned int) {param_name});\n"
        elif param_type in ["MPI_Aint *", "MPI_Count *"]:
            trace_code += f"  WRITE_TRACE(\"%llu:\", (long long unsigned int) *{param_name});\n"
        elif param_type == "MPI_Request *":
            trace_code += f"  WRITE_TRACE(\"%p:\", {param_name});\n"
        elif param_type == "MPI_Status *":
            trace_code += f"  WRITE_TRACE(\"%p:\", {param_name});\n"
        else:
            #print(f"Tracing for [{param_type}] not implemented! (appears in {function_name})")
            print(f"Tracing for [{param_type}] not implemented!")
        trace_code += "  }\n"
        return trace_code

    def generate_function_signature(self, cursor):
        return_type = cursor.result_type.spelling
        function_name = cursor.spelling
        parameters = []
        for param_cursor in cursor.get_children():
            if param_cursor.kind == clang.cindex.CursorKind.PARM_DECL:
                param_type = param_cursor.type.spelling
                param_name = param_cursor.spelling
                # If we use any square brackets, the param name comes before them, otherwise its type paramname.
                m = re.match("(.+?)(\[.*\])", param_type)
                if m:
                    parameters.append(f"{m.group(1)} {param_name}{m.group(2)}")
                else:
                    parameters.append(f"{param_type} {param_name}")
                self.types[str(param_type)] += [function_name]
        function_code  = f"{return_type} {function_name}({', '.join(parameters)})"
        return function_code


    def traverse_ast(self, node, depth=0, print_ast=False):
        if print_ast:
            print('  ' * depth + f'{node.kind} ({node.displayname})')
        if node.kind is clang.cindex.CursorKind.FUNCTION_DECL and re.match("MPI_.*", node.displayname) :
            self.nodes += [node]
        for child in node.get_children():
            self.traverse_ast(child, depth + 1, print_ast)


    def process_func(self, node):
        function_name = node.spelling
        if function_name in self.BLACKLISTED_FUNCTIONS:
            return
        self.outfile.write(self.generate_function_signature(node)+" {\n")
        self.outfile.write(self.generate_trace_code_prolog(node))
        self.outfile.write(self.generate_pmpi_call(node))
        self.outfile.write(self.generate_trace_code_epilog(node))
        self.outfile.write("  return retval;\n")
        self.outfile.write("}\n\n")

    def process_header(self, filename):
        clang.cindex.Config.set_library_path('/opt/homebrew/opt/llvm/lib')  # Update with the actual library path
        index = clang.cindex.Index.create()
        translation_unit = index.parse(filename)
        if not translation_unit:
            print("Error parsing the file.")
            return
        root_cursor = translation_unit.cursor
        self.traverse_ast(root_cursor)
        for node in self.nodes:
            self.process_func(node)

    def write_prolog(self):
        self.outfile.write("#include <mpi.h>\n")
        self.outfile.write("#include <assert.h>\n")
        self.outfile.write("#include <stdio.h>\n")
        self.outfile.write("\n")
        self.outfile.write("#define WRITE_TRACE(fmt, args...) printf(fmt, args)\n")
        self.outfile.write("#define COMMID_KEY 1234\n")
        self.outfile.write("#define WINID_KEY 2234\n")
        self.outfile.write("int next_commid = 0;\n")
        self.outfile.write("\n\n")

if __name__ == "__main__":
    codegen = AllprofCodegen()
    codegen.outfile = open("foo.c", "w")
    codegen.write_prolog()
    codegen.process_header("mpi.h")
    codegen.outfile.close()
    #for k in codegen.types:
    #    print(k + " appears in "+str(codegen.types[k]) +"\n")
