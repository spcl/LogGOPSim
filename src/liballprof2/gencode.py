#! /usr/bin/env python3

import re
import clang.cindex
import argparse
from collections import defaultdict
import yaml

class AllprofCodegen:

    def __init__(self):
        self.semantics = {}

    def parse_semantics(self, semfile):
        with open(semfile, "r") as stream:
            try:
                self.semantics = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def write_prolog(self, mode='c'):
        if mode == "c":
            self.outfile.write("#include <mpi.h>\n")
            self.outfile.write("#include <assert.h>\n")
            self.outfile.write("#include <stdio.h>\n")
            self.outfile.write("#include <string.h>\n")
            self.outfile.write("\n")
            self.outfile.write("#define WRITE_TRACE(fmt, args...) printf(fmt, args)\n")
            self.outfile.write("int COMMID_KEY = 1234;\n")
            self.outfile.write("int WINID_KEY = 2234;\n")
            self.outfile.write("int next_commid = 0;\n")
            # TODO move to MPI_Init!
            #self.outfile.write("MPI_Comm_create_keyval( MPI_NULL_COPY_FN, MPI_NULL_DELETE_FN, &COMMID_KEY, (void *)0 );\n")
            #self.outfile.write("MPI_Comm_set_attr(MPI_COMM_WORLD, COMMID_KEY, &next_commid);\n")
            #self.outfile.write("MPI_Win_create_keyval(MPI_NULL_COPY_FN, MPI_NULL_DELETE_FN, &WINID_KEY, (void *)0 );\n")
        
            self.outfile.write("\n\n")
        else:
            raise NotImplementedError(f"Mode {mode} not implemented!")
    
    def write_tracer_prolog(self, func, mode):
        """ Write a tracers prolog code, which writes starttime and function name to trace. Return value is expected in pmpi_retval!"""
        prolog_code = ""
        prolog_code += f"  {self.semantics[func]['return_type']} pmpi_retval;\n" 
        prolog_code += f"  WRITE_TRACE(\"%s\", \"{func}:\");\n"
        prolog_code += f"  WRITE_TRACE(\"%0.2f:\", PMPI_Wtime()*1e6);\n"
        self.outfile.write(prolog_code)

    def write_tracer_epilog(self, func, mode):
        """ Write a tracers epilog code, which writes endtime and returns pmpi_retval!"""
        code = ""
        code += f"  WRITE_TRACE(\"%0.2f\", PMPI_Wtime()*1e6);\n"
        code += f"  return pmpi_retval;\n"
        self.outfile.write(code)

    def write_pmpi_call(self, func, mode):
        """ Write the PMPI call. """
        sem = self.semantics[func]
        args = []
        for arg in self.semantics[func]['params']:
            args.append(arg['name'])
        argstr = ", ".join(args)
        self.outfile.write(f"  pmpi_retval = P{func}({argstr});\n")

    def split_type(self, typestr):
        """ Type descriptions like int[] cannot simply be prepended to an argument name foo, it must be int foo[]. This function seperates the base type and the [] part. """
        m = re.match("(.+?)(\[.*\])", typestr)
        if m:
            return (m.group(1), m.group(2))
        else:
            return (typestr, "")

    def tracer_for_simple_arg(self, name, typestr, func, sep=":"):
        if typestr.startswith("const "):
            typestr = typestr[6:]
        if typestr == "int":
            return f"WRITE_TRACE(\"%i{sep}\", {name});\n"
        if typestr == "int[3]":
            return f"WRITE_TRACE(\"[%i,%i,%i]{sep}\", {name}[0], {name}[1], {name}[2]);\n"
        if typestr == "char":
            return f"WRITE_TRACE(\"%c{sep}\", {name});\n"
        if typestr in ["MPI_Aint", "MPI_Count", "MPI_Offset"]:
            return f"WRITE_TRACE(\"%lli{sep}\", (long long int) {name});\n"
        elif typestr == "int *":
            return f"WRITE_TRACE(\"%i{sep}\", *{name});\n"
        elif typestr in ["MPI_Aint *", "MPI_Count *", "MPI_Offset *"] :
            return f"WRITE_TRACE(\"%lli{sep}\", (long long int) *{name});\n"
        elif typestr == "void *":
            return f"WRITE_TRACE(\"%p{sep}\", {name});\n"
        elif typestr.startswith("char *"):
            return f"WRITE_TRACE(\"%p{sep}\", {name});\n"
        elif typestr == "MPI_Comm":
            trace_code = ""
            # We set an info key in any function creating a new comm. These keys are NOT guranteed to be globally unique, only per comm_world rank!
            trace_code += "{\n"
            trace_code += f"  int comm_id, val_present, comm_rank, comm_size;\n"
            trace_code += f"  PMPI_Comm_get_attr({name}, COMMID_KEY, &comm_id, &val_present);\n"
            trace_code += f"  assert(val_present);\n"
            trace_code += f"  PMPI_Comm_rank({name}, &comm_rank);\n"
            trace_code += f"  PMPI_Comm_size({name}, &comm_size);\n"
            trace_code += f"  WRITE_TRACE(\"%i,%i,%i{sep}\", comm_id, comm_rank, comm_size);\n"
            trace_code += "}\n"
            return trace_code
        elif typestr == "MPI_Comm *":
            trace_code = ""
            # We set an info key in any function creating a new comm. These keys are NOT guranteed to be globally unique, only per comm_world rank!
            trace_code += "{\n"
            trace_code += f"  int comm_id, val_present, comm_rank, comm_size;\n"
            if ("new" in name) or ("graph" in name):
                trace_code += f"  PMPI_Comm_set_attr(*{name}, COMMID_KEY, &next_commid);"
                trace_code += f"  next_commid += 1;"
            trace_code += f"  PMPI_Comm_get_attr(*{name}, COMMID_KEY, &comm_id, &val_present);\n"
            trace_code += f"  assert(val_present);\n"
            trace_code += f"  PMPI_Comm_rank(*{name}, &comm_rank);\n"
            trace_code += f"  PMPI_Comm_size(*{name}, &comm_size);\n"
            trace_code += f"  WRITE_TRACE(\"%i,%i,%i{sep}\", comm_id, comm_rank, comm_size);\n"
            trace_code += "}\n"
            return trace_code
        elif typestr == "MPI_Win":
            trace_code = "{\n"
            trace_code += f"  int win_id, val_present;\n"
            trace_code += f"  PMPI_Win_get_attr({name}, WINID_KEY, &win_id, &val_present);\n"
            trace_code += f"  WRITE_TRACE(\"%i{sep}\", win_id);\n"
            trace_code += "}\n"
            return trace_code
        elif typestr == "MPI_Win *":
            # TODO actually set the info key in all functions creating a window
            trace_code = "{\n"
            trace_code += f"  int win_id, val_present;\n"
            trace_code += f"  PMPI_Win_get_attr(*{name}, WINID_KEY, &win_id, &val_present);\n"
            trace_code += f"  WRITE_TRACE(\"%i{sep}\", win_id);\n"
            trace_code += "}\n"
            return trace_code
        elif typestr == "MPI_Info":
            trace_code = f"  WRITE_TRACE(\"%llu{sep}\", (long long unsigned int) {name});\n"
            return trace_code
        elif typestr == "MPI_Request *": #maybe also find out if this is REQUEST_NULL?
            trace_code = f"  WRITE_TRACE(\"%p{sep}\", {name});\n"
        elif typestr == "MPI_Request": #maybe also find out if this is REQUEST_NULL?
            trace_code = f"  WRITE_TRACE(\"%p{sep}\", &{name});\n"
            return trace_code
        elif typestr == "MPI_Status *": #maybe also find out if this is STATUS_IGNORE?
            trace_code = f"  WRITE_TRACE(\"%p{sep}\", {name});\n"
        elif typestr == "MPI_Status": #maybe also find out if this is STATUS_IGNORE?
            trace_code = f"  WRITE_TRACE(\"%p{sep}\", &{name});\n"
            return trace_code
        elif typestr == "MPI_Datatype":
            trace_code = "{\n"
            trace_code += f"  int ddtsize;\n"
            trace_code += f"  PMPI_Type_size({name}, &ddtsize);\n"
            trace_code += f"  WRITE_TRACE(\"%i{sep}\", ddtsize);\n"
            trace_code += "}\n"
        elif typestr == "MPI_Datatype *":
            trace_code = "{\n"
            trace_code += f"  int ddtsize;\n"
            trace_code += f"  PMPI_Type_size(*{name}, &ddtsize);\n"
            trace_code += f"  WRITE_TRACE(\"%i{sep}\", ddtsize);\n"
            trace_code += "}\n"
            return trace_code
        elif typestr.endswith("_function *"):
            return f"WRITE_TRACE(\"%p{sep}\", {name});\n"
        elif typestr == "MPI_Group":
            trace_code = f"WRITE_TRACE(\"%i\", PMPI_Group_c2f({name}));"
        elif typestr == "MPI_Group *":
            trace_code = f"WRITE_TRACE(\"%i\", PMPI_Group_c2f(*{name}));"
        elif typestr == "MPI_File":
            trace_code = f"WRITE_TRACE(\"%i\", PMPI_File_c2f({name}));"
        elif typestr == "MPI_File *":
            trace_code = f"WRITE_TRACE(\"%i\", PMPI_File_c2f(*{name}));"
        elif typestr == "MPI_Info":
            trace_code = f"WRITE_TRACE(\"%i\", PMPI_Info_c2f({name}));"
        elif typestr == "MPI_Info *":
            trace_code = f"WRITE_TRACE(\"%i\", PMPI_Info_c2f(*{name}));"
        elif typestr == "MPI_Op":
            trace_code = f"WRITE_TRACE(\"%i\", PMPI_Op_c2f({name}));"
        elif typestr == "MPI_Op *":
            trace_code = f"WRITE_TRACE(\"%i\", PMPI_Op_c2f(*{name}));"
        elif typestr == "MPI_Errhandler":
            trace_code = f"WRITE_TRACE(\"%i\", PMPI_Errhandler_c2f({name}));"
        elif typestr == "MPI_Errhandler *":
            trace_code = f"WRITE_TRACE(\"%i\", PMPI_Errhandler_c2f(*{name}));"
        elif typestr == "MPI_Message":
            trace_code = f"WRITE_TRACE(\"%i\", PMPI_Message_c2f({name}));"
        elif typestr == "MPI_Message *":
            trace_code = f"WRITE_TRACE(\"%i\", PMPI_Message_c2f(*{name}));"
        else:
            print(f"{typestr} tracer not implemmented (appears in {func})")
        return ""

    def write_argument_tracers(self, func, mode):
        #TODO do not emit the same prolog multiple times
        for sem_param in self.semantics[func]['params']:
            #print(sem_param)
            if ('elem_count' in sem_param) and sem_param['elem_count'] is not None:
                # this argument contains multiple elements
                if ('prolog_elem_count' in sem_param) and (sem_param['prolog_elem_count'] is not None):
                    self.outfile.write("{\n")
                    self.outfile.write(f"  {sem_param['prolog_elem_count']}\n")
                self.outfile.write(f"  for (int trace_elem_idx=0; trace_elem_idx<{sem_param['elem_count']}; trace_elem_idx++) "+"{\n")
                # emit the tracer for the simplified arg
                name = sem_param['name'] + "[trace_elem_idx]"
                basetype, brackets = self.split_type(sem_param['type'])
                typestr = basetype + brackets[2:]
                code = self.tracer_for_simple_arg(name, typestr, func, sep=";")
                self.outfile.write("    "+code)
                self.outfile.write("  }\n")
                if ('prolog_elem_count' in sem_param) and (sem_param['prolog_elem_count'] is not None):
                    self.outfile.write("WRITE_TRACE(\"\%s\", \":\");\n")
                    self.outfile.write("}\n")
                if ('elem_count' in sem_param) and (sem_param['prolog_elem_count'] is None):
                    self.outfile.write(f"WRITE_TRACE(\"%p:\", {sem_param['name']});\n")
            else:
                code = self.tracer_for_simple_arg(sem_param['name'], sem_param['type'], func, sep=":")
                self.outfile.write("  " + code)

    def produce_tracers(self, mode='c'):
        for func in self.semantics:
            delay_pmpi = False # usually we write the trace after the pmpi call, however, if the function frees some of its arguments we want to do it before
            param_signatures = []
            for param in self.semantics[func]['params']:
                type_prefix, type_suffix = self.split_type(param['type'])
                param_signatures.append(f"{type_prefix} {param['name']}{type_suffix}")
            if func.endswith("_free") or ("_delete_" in func):
                delay_pmpi = True
            params = ", ".join(param_signatures) 
            self.outfile.write(f"{self.semantics[func]['return_type']} {func} ({params})"+" {\n")
            self.write_tracer_prolog(func, mode)
            if not delay_pmpi:
                self.write_pmpi_call(func, mode)
            self.write_argument_tracers(func, mode)
            if delay_pmpi:
                self.write_pmpi_call(func, mode)
            self.write_tracer_epilog(func, mode)
            self.outfile.write("}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='liballprof_gencode',
                    description='Generates wrappers for the MPI functions present in the supplied MPI header file. The wrappers output in liballprof2 trace format.',
                    epilog='')
    parser.add_argument('-s', '--semantics-file',       default='mpi_sem.yml',              help="Name of the file that specifies the tracer semantics (default: mpi-sem.yml)")
    parser.add_argument('-c', '--c-output-file',        default="mpi_c_wrapper.c",          help="Name of the generated C file (default: mpi_c_wrapper.c)")
    parser.add_argument('-f', '--fortran-output-file',  default="mpi_f_wrapper.f90",        help="Name of the generated FORTRAN file (default: mpi_f_wrapper.f90)")
    args = parser.parse_args()

    codegen = AllprofCodegen()
    codegen.parse_semantics(args.semantics_file)
    codegen.outfile = open(args.c_output_file, "w")
    codegen.write_prolog(mode='c')
    codegen.produce_tracers(mode='c')
    codegen.outfile.close()

