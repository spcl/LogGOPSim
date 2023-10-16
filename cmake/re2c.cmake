macro (find_re2c)
  if (NOT RE2C_EXECUTABLE)
    find_program (RE2C_EXECUTABLE re2c)
    if (NOT RE2C_EXECUTABLE)
      message (FATAL_ERROR "re2c not found. Aborting...")
    endif ()
  endif ()
endmacro ()

macro (add_re2c_files _basename)
  find_re2c()

  set (_re2c_in  ${CMAKE_CURRENT_SOURCE_DIR}/${_basename}.re)
  set (_re2c_out ${CMAKE_CURRENT_SOURCE_DIR}/${_basename}.cpp)

  get_filename_component(_basepath ${_basename} DIRECTORY)
  get_filename_component(_basefile ${_basename} NAME)

  add_custom_command (
    OUTPUT ${_re2c_out}
    COMMAND re2c  ${_re2c_in} -o ${_re2c_out}
    DEPENDS ${_re2c_in}
#    BYPRODUCTS
    COMMENT "Generating re2c parser code ..."
    VERBATIM
    )

endmacro (add_re2c_files)
