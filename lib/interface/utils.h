
#include <../openjpeg/src/lib/openjp2/openjpeg.h>

// Size of the buffer for the input/output streams
#define BUFFER_SIZE OPJ_J2K_STREAM_CHUNK_SIZE

#ifndef _PYOPJ_UTILS_H_
#define _PYOPJ_UTILS_H_

  // BinaryIO.tell()
  extern Py_ssize_t py_tell(PyObject *stream);

  // BinaryIO.read()
  extern OPJ_SIZE_T py_read(void *destination, OPJ_SIZE_T nr_bytes, void *fd);

  // BinaryIO.seek()
  extern OPJ_BOOL py_seek(Py_ssize_t offset, void *stream, int whence);

  // BinaryIO.seek(offset, SEEK_SET)
  extern OPJ_BOOL py_seek_set(OPJ_OFF_T offset, void *stream);

  // BinaryIO.seek(offset, SEEK_CUR)
  extern OPJ_OFF_T py_skip(OPJ_OFF_T offset, void *stream);

  // len(BinaryIO)
  extern OPJ_UINT64 py_length(PyObject *stream);

  // BinaryIO.write()
  extern OPJ_SIZE_T py_write(void *src, OPJ_SIZE_T nr_bytes, void *dst);

  // Log a message to the Python logger
  extern void py_log(char *name, char *log_level, const char *log_msg);


#endif
