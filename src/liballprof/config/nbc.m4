AC_DEFUN([NBC_CHECK_NBC],[
	AC_ARG_WITH(nbc,
        AC_HELP_STRING([--with-nbc], [search for libNBC in the given path]))
        if test "x${with_nbc}" != "xno"; then
          if test "x${with_nbc}" != "x"; then
            AC_MSG_NOTICE([*** checking for libNBC ***])
            if test -f "${with_nbc}/lib/libnbc.a"; then
                LDFLAGS="${LDFLAGS} -L${with_nbc}/lib"
                CPPFLAGS="${CPPFLAGS} -I${with_nbc}/include"
            fi
            if test -f "${with_nbc}/.libs/libnbc.a"; then
                LDFLAGS="${LDFLAGS} -L${with_nbc}/.libs/"
                CPPFLAGS="${CPPFLAGS} -I${with_nbc}"
            fi
            if test -f "../src/.libs/libnbc.a"; then
                LDFLAGS="${LDFLAGS} -L../src/.libs/"
                CPPFLAGS="${CPPFLAGS} -I../src"
            fi
            AC_CHECK_LIB([nbc], [NBC_Ibcast],,AC_MSG_ERROR(libNBC not found))
            AC_CHECK_HEADER([nbc.h],,AC_MSG_ERROR(nbc.h not found))
            AC_DEFINE(HAVE_NBC, 1, enables NBC code)
          fi  
        fi
   ]
)
