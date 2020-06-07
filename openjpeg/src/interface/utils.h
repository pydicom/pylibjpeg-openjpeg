#include <../openjpeg/src/lib/openjp2/openjpeg.h>

extern int decode(PyObject *inArray, unsigned char* out);
extern Py_ssize_t read_data(PyObject *fd, char* destination, Py_ssize_t nr_bytes);

static OPJ_SIZE_T j2k_read(void *p_buffer, OPJ_SIZE_T p_nb_bytes, void *p_user_data);
