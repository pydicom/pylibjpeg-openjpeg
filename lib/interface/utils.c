
#include "Python.h"
#include "utils.h"
#include <../openjpeg/src/lib/openjp2/openjpeg.h>


// Python logging
void py_log(char *name, char *log_level, const char *log_msg)
{
    /* Log `log_msg` to the Python logger `name`.

    Parameters
    ----------
    name
        The name of the logger (i.e. logger.getLogger(name))
    log_level
        The log level to use DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_msg
        The message to use.
    */
    static PyObject *module = NULL;
    static PyObject *logger = NULL;
    static PyObject *msg = NULL;
    static PyObject *lib_name = NULL;

    // import logging
    module = PyImport_ImportModuleNoBlock("logging");
    if (module == NULL) {
        return;
    }

    // logger = logging.getLogger(lib_name)
    lib_name = Py_BuildValue("s", name);
    logger = PyObject_CallMethod(module, "getLogger", "O", lib_name);

    // Create Python str and remove trailing newline
    msg = Py_BuildValue("s", log_msg);
    msg = PyObject_CallMethod(msg, "strip", NULL);

    // logger.debug(msg), etc
    if (strcmp(log_level, "DEBUG") == 0) {
        PyObject_CallMethod(logger, "debug", "O", msg);
    } else if (strcmp(log_level, "INFO") == 0) {
        PyObject_CallMethod(logger, "info", "O", msg);
    } else if (strcmp(log_level, "WARNING") == 0) {
        PyObject_CallMethod(logger, "warning", "O", msg);
    } else if (strcmp(log_level, "ERROR") == 0) {
        PyObject_CallMethod(logger, "error", "O", msg);
    } else if (strcmp(log_level, "CRITICAL") == 0) {
        PyObject_CallMethod(logger, "critical", "O", msg);
    }

    Py_DECREF(msg);
    Py_DECREF(lib_name);
}

// Python BinaryIO methods
Py_ssize_t py_tell(PyObject *stream)
{
    /* Return the current position of the `stream`.

    Parameters
    ----------
    stream : PyObject *
        The Python stream object to use (must have a ``tell()`` method).

    Returns
    -------
    Py_ssize_t
        The current position in the `stream`.
    */
    PyObject *result;
    Py_ssize_t location;

    result = PyObject_CallMethod(stream, "tell", NULL);
    location = PyLong_AsSsize_t(result);

    Py_DECREF(result);

    //printf("py_tell(): %u\n", location);
    return location;
}


OPJ_SIZE_T py_read(void *destination, OPJ_SIZE_T nr_bytes, void *fd)
{
    /* Read `nr_bytes` from Python object `fd` and copy it to `destination`.

    Parameters
    ----------
    destination : void *
        The object where the read data will be copied.
    nr_bytes : OPJ_SIZE_T
        The number of bytes to be read.
    fd : PyObject *
        The Python file-like to read the data from (must have a ``read()``
        method).

    Returns
    -------
    OPJ_SIZE_T
        The number of bytes read or -1 if reading failed or if trying to read
        while at the end of the data.
    */
    PyObject* result;
    char* buffer;
    Py_ssize_t length;
    int bytes_result;

    // Py_ssize_t: signed int samed size as size_t
    // fd.read(nr_bytes), "k" => C unsigned long int to Python int
    result = PyObject_CallMethod(fd, "read", "n", nr_bytes);
    // Returns the null-terminated contents of `result`
    // `length` is Py_ssize_t *
    // `buffer` is char **
    // If `length` is NULL, returns -1
    bytes_result = PyBytes_AsStringAndSize(result, &buffer, &length);

    // `length` is NULL
    if (bytes_result == -1)
        goto error;

    // More bytes read then asked for
    if (length > (long)(nr_bytes))
        goto error;

    // Convert `length` to OPJ_SIZE_T - shouldn't have negative lengths
    if (length < 0)
        goto error;

    OPJ_SIZE_T len_size_t = (OPJ_SIZE_T)(length);

    // memcpy(void *dest, const void *src, size_t n)
    memcpy(destination, buffer, len_size_t);

    //printf("py_read(): %u bytes asked, %u bytes read\n", nr_bytes, len_size_t);

    Py_DECREF(result);
    return len_size_t ? len_size_t: (OPJ_SIZE_T)-1;

error:
    Py_DECREF(result);
    return -1;
}


OPJ_BOOL py_seek(Py_ssize_t offset, void *stream, int whence)
{
    /* Change the `stream` position to the given `offset` from `whence`.

    Parameters
    ----------
    offset : OPJ_OFF_T
        The offset relative to `whence`.
    stream : PyObject *
        The Python stream object to seek (must have a ``seek()`` method).
    whence : int
        0 for SEEK_SET, 1 for SEEK_CUR, 2 for SEEK_END

    Returns
    -------
    OPJ_TRUE : OBJ_BOOL
    */
    // Python and C; SEEK_SET is 0, SEEK_CUR is 1 and SEEK_END is 2
    // stream.seek(nr_bytes, whence),
    // k: convert C unsigned long int to Python int
    // i: convert C int to a Python integer
    PyObject *result;
    result = PyObject_CallMethod(stream, "seek", "ni", offset, whence);
    Py_DECREF(result);

    //printf("py_seek(): offset %u bytes from %u\n", offset, whence);

    return OPJ_TRUE;
}


OPJ_BOOL py_seek_set(OPJ_OFF_T offset, void *stream)
{
    /* Change the `stream` position to the given `offset` from SEEK_SET.

    Parameters
    ----------
    offset : OPJ_OFF_T
        The offset relative to SEEK_SET.
    stream : PyObject *
        The Python stream object to seek (must have a ``seek()`` method).

    Returns
    -------
    OPJ_TRUE : OBJ_BOOL
    */
    return py_seek(offset, stream, SEEK_SET);
}


OPJ_OFF_T py_skip(OPJ_OFF_T offset, void *stream)
{
    /* Change the `stream` position by `offset` from SEEK_CUR and return the
    number of skipped bytes.

    Parameters
    ----------
    offset : OPJ_OFF_T
        The offset relative to SEEK_CUR.
    stream : PyObject *
        The Python stream object to seek (must have a ``seek()`` method).

    Returns
    -------
    int
        The number of bytes skipped
    */
    off_t initial;
    initial = py_tell(stream);

    py_seek(offset, stream, SEEK_CUR);

    off_t current;
    current = py_tell(stream);

    return current - initial;
}


OPJ_UINT64 py_length(PyObject * stream)
{
    /* Return the total length of the `stream`.

    Parameters
    ----------
    stream : PyObject *
        The Python stream object (must have ``seek()`` and ``tell()`` methods).

    Returns
    -------
    OPJ_UINT64
        The total length of the `stream`.
    */
    OPJ_OFF_T input_length = 0;

    py_seek(0, stream, SEEK_END);
    input_length = (OPJ_OFF_T)py_tell(stream);
    py_seek(0, stream, SEEK_SET);

    return (OPJ_UINT64)input_length;
}


OPJ_SIZE_T py_write(void *src, OPJ_SIZE_T nr_bytes, void *dst)
{
    /* Write `nr_bytes` from `src` to the Python object `dst`

    Parameters
    ----------
    src : void *
        The source of data.
    nr_bytes : OPJ_SIZE_T
        The length of the data to be written.
    dst : void *
        Where the data should be written to. Should be BinaryIO.

    Returns
    -------
    OBJ_SIZE_T
        The number of bytes written to dst.
    */
    PyObject *result;
    PyObject *bytes_object;

    // Create bytes object from `src` (of length `bytes`)
    bytes_object = PyBytes_FromStringAndSize(src, nr_bytes);
    // Use the bytes object to extend our dst using write(bytes_object)
    // 'O' means pass the Python object untouched
    result = PyObject_CallMethod(dst, "write", "O", bytes_object);

    Py_DECREF(bytes_object);
    Py_DECREF(result);

    return nr_bytes;
}
