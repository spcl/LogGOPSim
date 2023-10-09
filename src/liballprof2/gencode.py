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
            self.outfile.write("int COMMID_KEY;\n")
            self.outfile.write("int WINID_KEY;\n")
            self.outfile.write("int next_commid = 0;\n\n")
            self.outfile.write("void tracer_init(void) {\n")
            self.outfile.write("  PMPI_Comm_create_keyval( MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &COMMID_KEY, (void *)0 );\n")
            self.outfile.write("  PMPI_Comm_set_attr(MPI_COMM_WORLD, COMMID_KEY, &next_commid);\n")
            self.outfile.write("  PMPI_Win_create_keyval(MPI_WIN_NULL_COPY_FN, MPI_WIN_NULL_DELETE_FN, &WINID_KEY, (void *)0 );\n")
            self.outfile.write("}\n")
        
            self.outfile.write("\n\n")
        elif mode == 'fortran':
            self.outfile.write("include \"mpif.h\"\n\n")
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
        code += f"  WRITE_TRACE(\"%0.2f\\n\", PMPI_Wtime()*1e6);\n"
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
        
    def get_basetype(self, typestr):
        """ Remove any attribute such as const, *, [] from typestr """
        m = re.match("^(.+?)(\[.*)", typestr)
        if m:
            typestr = m.group(1)
        elems = typestr.split(" ")
        if elems[-1] == "*":
            elems = elems[:-1]
        return elems[-1]
        
    def is_inttype(self, typestr):
        """ Check if typestr is a type which we can cast to a long long int without loosing precision """
        if typestr in ['int', 'MPI_Aint', 'MPI_Offset', 'MPI_Count']:
            return True
        return False
    
    def is_ptr(self, typestr):
        if ("*" in typestr) or ("[" in typestr):
            return True
        return False
    
    def trace_inttype(self, name, typestr, basetype, is_ptr, sep):
        deref = ""
        if is_ptr:
            deref = "*"
        return f"WRITE_TRACE(\"%lli{sep}\", (long long int) {deref}({name}));"

    def is_mpiobj(self, typestr):
        if typestr in ['MPI_Comm', 'MPI_Group', 'MPI_Win', 
                       'MPI_Info', 'MPI_Message', 'MPI_Op', 
                       'MPI_Request', 'MPI_Status', 'MPI_Datatype', 
                       'MPI_File', 'MPI_Errhandler']:
            return True
        return False

    def trace_mpiobj(self, name, typestr, basetype, is_ptr, sep):
        deref = ""
        ref = "&"
        if is_ptr:
            deref = "*"
            ref = ""
        if basetype == "MPI_Datatype": #the mpi forum really values consistency - except when they don't :)
            basetype = "MPI_Type"
        if basetype == "MPI_Status": # at least here it makes sense :)
            if is_ptr: # apparently c2f has a prolem with MPI_STATUSES_IGNORE :( 
                return f"if ({name} == MPI_STATUSES_IGNORE) {{WRITE_TRACE(\"%lli\", (long long int) MPI_STATUSES_IGNORE);}} else {{MPI_Fint fstatus; PMPI_Status_c2f({ref}{name}, &fstatus); WRITE_TRACE(\"%lli{sep}\", (long long int) fstatus);}}"
            else:
                return f"{{MPI_Fint fstatus; PMPI_Status_c2f({ref}{name}, &fstatus); WRITE_TRACE(\"%lli{sep}\", (long long int) fstatus);}}"
        return f"WRITE_TRACE(\"%lli{sep}\", (long long int) P{basetype}_c2f({deref}{name}));"
        

    def tracer_for_simple_arg(self, name, typestr, func, sep=":"):
        # strip const, shouldn't make a difference how we trace
        if typestr.startswith("const "):
            typestr = typestr[6:]
        # handle this special case first
        if typestr == "int[3]":
            return f"WRITE_TRACE(\"[%i,%i,%i]{sep}\", {name}[0], {name}[1], {name}[2]);\n"
        basetype = self.get_basetype(typestr)
        is_ptr = self.is_ptr(typestr)
        if self.is_inttype(basetype):
            return self.trace_inttype(name, typestr, basetype, is_ptr, sep)
        elif self.is_mpiobj(basetype):
            return self.trace_mpiobj(name, typestr, basetype, is_ptr, sep)
        elif typestr == "char":
            return f"WRITE_TRACE(\"%c{sep}\", {name});\n"
        elif typestr == "void *":
            return f"WRITE_TRACE(\"%p{sep}\", {name});\n"
        elif typestr.startswith("char *"):
            return f"WRITE_TRACE(\"%p{sep}\", {name});\n"
        elif typestr.endswith("_function *"):
            return f"WRITE_TRACE(\"%p{sep}\", {name});\n"
        else:
            print(f"{typestr} tracer not implemmented (appears in {func}, basetype is {basetype})")
        return ""

    def write_argument_tracers(self, func, mode):
        # collect all needed prologs (code that we need to decide how many elemnts are in an array, like getting comm size)
        prologs = []
        for sem_param in self.semantics[func]['params']:
            if ('prolog_elem_count' in sem_param) and (sem_param['prolog_elem_count'] is not None):
                prologs.append(sem_param['prolog_elem_count'])
        prologs = list(set(prologs))
        if len(prologs) > 0:
            self.outfile.write("\n".join(prologs)+"\n//end of prologs\n")

        for sem_param in self.semantics[func]['params']:
            #print(sem_param)
            if ('elem_count' in sem_param):
                # this argument contains multiple elements
                # let the user configure at runtime if elements need to be traced (replace 0 with env-bases expr)
                elem_count_expr = "0"
                if ('elem_count' in sem_param) and (sem_param['elem_count'] is not None):
                    elem_count_expr = sem_param['elem_count']
                self.outfile.write(f"  WRITE_TRACE(\"%p,%i[\", (void*) {sem_param['name']}, (int) {elem_count_expr});\n")
                self.outfile.write(f"  if (0) {{  }} else {{ \n")
                self.outfile.write(f"    for (int trace_elem_idx=0; trace_elem_idx<{elem_count_expr}; trace_elem_idx++) "+"{\n")
                
                # emit the tracer for the simplified arg, use ; to seperate elems
                name = sem_param['name'] + "[trace_elem_idx]"
                basetype, brackets = self.split_type(sem_param['type'])
                typestr = basetype + brackets[2:]
                code = self.tracer_for_simple_arg(name, typestr, func, sep=";")
                self.outfile.write("    "+code)
                self.outfile.write("  }\n")  
                self.outfile.write("  WRITE_TRACE(\"]%s\", \":\");\n")
                self.outfile.write("}\n")               
            else:
                code = self.tracer_for_simple_arg(sem_param['name'], sem_param['type'], func, sep=":")
                self.outfile.write("  " + code)

    def produce_tracers(self, mode='c'):
        for func in self.semantics:
            delay_pmpi = False # usually we write the trace after the pmpi call, however, if the function frees some of its arguments we want to do it before
            if func.endswith("_free") or ("_delete_" in func):
                delay_pmpi = True
            param_signatures = []
            for param in self.semantics[func]['params']:
                if mode == 'c':
                    type_prefix, type_suffix = self.split_type(param['type'])
                    param_signatures.append(f"{type_prefix} {param['name']}{type_suffix}")
                if mode == 'fortran':
                    param_signatures.append(f"int* {param['name']}")
            params = ", ".join(param_signatures)
            if mode == 'c':
                self.outfile.write(f"{self.semantics[func]['return_type']} {func} ({params})"+" {\n")
            if mode == 'fortran':
                self.outfile.write(f"void FortranCInterface_GLOBAL({func},{func})({params})"+" {\n")
            self.write_tracer_prolog(func, mode)
            if not delay_pmpi:
                self.write_pmpi_call(func, mode)
            if func == 'MPI_Init':
                self.outfile.write("tracer_init();\n")
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
    parser.add_argument('-c', '--c-output-file',        default="mpi_c_wrapper.c",          help="Name of the generated C wrapper file (default: mpi_c_wrapper.c)")
    parser.add_argument('-f', '--fortran-output-file',  default="mpi_f_wrapper.c",          help="Name of the generated FORTRAN wrapper file (default: mpi_f_wrapper.c)")
    args = parser.parse_args()

    codegen = AllprofCodegen()
    codegen.parse_semantics(args.semantics_file)
    codegen.outfile = open(args.c_output_file, "w")
    codegen.write_prolog(mode='c')
    codegen.produce_tracers(mode='c')
    codegen.outfile.close()
    codegen.outfile = open(args.fortran_output_file, "w")
    codegen.write_prolog(mode='fortran')
    codegen.produce_tracers(mode='fortran')
    codegen.outfile.close()

