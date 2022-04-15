/* auxiliary_ctools.i */

%module auxiliary_ctools
%include "cpointer.i"
%pointer_functions(long, longP)
%include "carrays.i"
%array_class(long, longArray);


%typemap(out) long *single_numset %{
  $result = PyList_New(500); // use however you know the size here
  for (int i = 0; i < 500; ++i) {
    PyList_SetItem($result, i, PyInt_FromLong($1[i]));
  }
  //delete $1; // Important to avoid a leak since you called new
%}

%{
#include "auxiliary_ctools.h"
%}


long* single_numset(const char* seq, int q);