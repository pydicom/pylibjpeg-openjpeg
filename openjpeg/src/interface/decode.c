/*

py_read, py_seek, py_skip and py_tell are adapted from
Pillow/src/libimaging.codec_fd.c which is licensed under the PIL Software
License:

The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh

Pillow is the friendly PIL fork. It is

    Copyright © 2010-2020 by Alex Clark and contributors

Like PIL, Pillow is licensed under the open source PIL Software License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify, and distribute this software and its
associated documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

------------------------------------------------------------------------------

Bits and pieces of the rest are adapted from
openjpeg/src/bin/jp2/opj_decompress.c which is licensed under the 2-clause BSD
license (see the main LICENSE file).
*/

#include "Python.h"
#include <stdlib.h>
#include <stdio.h>
#include <../openjpeg/src/lib/openjp2/openjpeg.h>


// Size of the buffer for the input stream
#define BUFFER_SIZE OPJ_J2K_STREAM_CHUNK_SIZE


const char * OpenJpegVersion(void)
{
    /* Return the openjpeg version as char array

    Returns
    -------
    char *
        The openjpeg version as MAJOR.MINOR.PATCH
    */
    return opj_version();
}


// parameters for decoding
typedef struct opj_decompress_params {
    /** core library parameters */
    opj_dparameters_t core;

    /** input file format 0: J2K, 1: JP2, 2: JPT */
    int decod_format;

    /** Decoding area left boundary */
    OPJ_UINT32 DA_x0;
    /** Decoding area right boundary */
    OPJ_UINT32 DA_x1;
    /** Decoding area up boundary */
    OPJ_UINT32 DA_y0;
    /** Decoding area bottom boundary */
    OPJ_UINT32 DA_y1;
    /** Verbose mode */
    OPJ_BOOL m_verbose;

    /** tile number of the decoded tile */
    OPJ_UINT32 tile_index;
    /** Nb of tile to decode */
    OPJ_UINT32 nb_tile_to_decode;

    /* force output colorspace to RGB */
    //int force_rgb;
    /** number of threads */
    //int num_threads;
    /* Quiet */
    //int quiet;
    /** number of components to decode */
    OPJ_UINT32 numcomps;
    /** indices of components to decode */
    OPJ_UINT32* comps_indices;
} opj_decompress_parameters;


// Python stream methods
static Py_ssize_t py_tell(PyObject *stream)
{
    /* Return the current position of the `stream`.

    Parameters
    ----------
    stream : PyObject *
        The Python stream object to use (must have a ``tell()`` method).

    Returns
    -------
    Py_ssize_t
        The new position in the `stream`.
    */
    PyObject *result;
    Py_ssize_t location;

    result = PyObject_CallMethod(stream, "tell", NULL);
    location = PyLong_AsSsize_t(result);

    Py_DECREF(result);

    //printf("py_tell(): %u\n", location);
    return location;
}


static OPJ_SIZE_T py_read(void *destination, OPJ_SIZE_T nr_bytes, void *fd)
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


static OPJ_BOOL py_seek(Py_ssize_t offset, void *stream, int whence)
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
    // fd.seek(nr_bytes),
    // k: convert C unsigned long int to Python int
    // i: convert C int to a Python integer
    PyObject *result;
    result = PyObject_CallMethod(stream, "seek", "ni", offset, whence);
    Py_DECREF(result);

    //printf("py_seek(): offset %u bytes from %u\n", offset, whence);

    return OPJ_TRUE;
}


static OPJ_BOOL py_seek_set(OPJ_OFF_T offset, void *stream)
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


static OPJ_OFF_T py_skip(OPJ_OFF_T offset, void *stream)
{
    /* Change the `stream` position by `offset` from SEEK_CUR and return the
    new position.

    Parameters
    ----------
    offset : OPJ_OFF_T
        The offset relative to SEEK_CUR.
    stream : PyObject *
        The Python stream object to seek (must have a ``seek()`` method).

    Returns
    -------
    off_t
        The new position in the `stream`.
    */
    py_seek(offset, stream, SEEK_CUR);

    off_t pos;
    pos = py_tell(stream);

    return pos ? pos : (OPJ_OFF_T) -1;
}


static OPJ_UINT64 py_length(PyObject * stream)
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


static void set_default_parameters(opj_decompress_parameters* parameters)
{
    if (parameters)
    {
        memset(parameters, 0, sizeof(opj_decompress_parameters));

        /* default decoding parameters (command line specific) */
        parameters->decod_format = -1;
        /* default decoding parameters (core) */
        opj_set_default_decoder_parameters(&(parameters->core));
    }
}


static void destroy_parameters(opj_decompress_parameters* parameters)
{
    if (parameters)
    {
        free(parameters->comps_indices);
        parameters->comps_indices = NULL;
    }
}


typedef struct JPEG2000Parameters {
    OPJ_UINT32 columns;  // width in pixels
    OPJ_UINT32 rows;  // height in pixels
    OPJ_COLOR_SPACE colourspace;  // the colour space
    OPJ_UINT32 nr_components;  // number of components
    OPJ_UINT32 precision;  // precision of the components (in bits)
    unsigned int is_signed;  // 0 for unsigned, 1 for signed
    OPJ_UINT32 nr_tiles;  // number of tiles
} j2k_parameters_t;


extern int GetParameters(PyObject* fd, int codec_format, j2k_parameters_t *output)
{
    /* Decode a JPEG 2000 header for the image meta data.

    Parameters
    ----------
    fd : PyObject *
        The Python stream object containing the JPEG 2000 data to be decoded.
    codec_format : int
        The format of the JPEG 2000 data, one of:
        * ``0`` - OPJ_CODEC_J2K : JPEG-2000 codestream
        * ``1`` - OPJ_CODEC_JPT : JPT-stream (JPEG 2000, JPIP)
        * ``2`` - OPJ_CODEC_JP2 : JP2 file format
    output : j2k_parameters_t *
        The struct where the parameters will be stored.

    Returns
    -------
    int
        The exit status, 0 for success, failure otherwise.
    */
    // J2K stream
    opj_stream_t *stream = NULL;
    // struct defining image data and characteristics
    opj_image_t *image = NULL;
    // J2K codec
    opj_codec_t *codec = NULL;
    // Setup decompression parameters
    opj_decompress_parameters parameters;
    set_default_parameters(&parameters);

    int error_code = EXIT_FAILURE;

    // Creates an abstract input stream; allocates memory
    stream = opj_stream_create(BUFFER_SIZE, OPJ_TRUE);

    if (!stream)
    {
        // Failed to create the input stream
        error_code = 1;
        goto failure;
    }

    // Functions for the stream
    opj_stream_set_read_function(stream, py_read);
    opj_stream_set_skip_function(stream, py_skip);
    opj_stream_set_seek_function(stream, py_seek_set);
    opj_stream_set_user_data(stream, fd, NULL);
    opj_stream_set_user_data_length(stream, py_length(fd));

    codec = opj_create_decompress(codec_format);

    /* Setup the decoder parameters */
    if (!opj_setup_decoder(codec, &(parameters.core)))
    {
        // failed to setup the decoder
        error_code = 2;
        goto failure;
    }

    /* Read the main header of the codestream and if necessary the JP2 boxes*/
    if (!opj_read_header(stream, codec, &image))
    {
        // failed to read the header
        error_code = 3;
        goto failure;
    }

    output->colourspace = image->color_space;
    output->columns = image->x1;
    output->rows = image->y1;
    output->nr_components = image->numcomps;
    output->precision = (int)image->comps[0].prec;
    output->is_signed = (int)image->comps[0].sgnd;
    output->nr_tiles = parameters.nb_tile_to_decode;

    destroy_parameters(&parameters);
    opj_destroy_codec(codec);
    opj_image_destroy(image);
    opj_stream_destroy(stream);

    return EXIT_SUCCESS;

    failure:
        destroy_parameters(&parameters);
        if (codec)
            opj_destroy_codec(codec);
        if (image)
            opj_image_destroy(image);
        if (stream)
            opj_stream_destroy(stream);

        return error_code;
}


extern int Decode(PyObject* fd, unsigned char *out, int codec_format)
{
    /* Decode JPEG 2000 data.

    Parameters
    ----------
    fd : PyObject *
        The Python stream object containing the JPEG 2000 data to be decoded.
    out : unsigned char *
        The numpy ndarray of uint8 where the decoded image data will be written
    codec_format : int
        The format of the JPEG 2000 data, one of:
        * ``0`` - OPJ_CODEC_J2K : JPEG-2000 codestream
        * ``1`` - OPJ_CODEC_JPT : JPT-stream (JPEG 2000, JPIP)
        * ``2`` - OPJ_CODEC_JP2 : JP2 file format

    Returns
    -------
    int
        The exit status, 0 for success, failure otherwise.
    */
    // J2K stream
    opj_stream_t *stream = NULL;
    // struct defining image data and characteristics
    opj_image_t *image = NULL;
    // J2K codec
    opj_codec_t *codec = NULL;
    // Setup decompression parameters
    opj_decompress_parameters parameters;
    set_default_parameters(&parameters);
    // Array of pointers to the first element of each component
    int **p_component = NULL;

    int error_code = EXIT_FAILURE;

    // Creates an abstract input stream; allocates memory
    stream = opj_stream_create(BUFFER_SIZE, OPJ_TRUE);

    if (!stream)
    {
        // Failed to create the input stream
        error_code = 1;
        goto failure;
    }

    // Functions for the stream
    opj_stream_set_read_function(stream, py_read);
    opj_stream_set_skip_function(stream, py_skip);
    opj_stream_set_seek_function(stream, py_seek_set);
    opj_stream_set_user_data(stream, fd, NULL);
    opj_stream_set_user_data_length(stream, py_length(fd));

    //opj_set_error_handler(codec, j2k_error, 00);

    codec = opj_create_decompress(codec_format);

    /* Setup the decoder parameters */
    if (!opj_setup_decoder(codec, &(parameters.core)))
    {
        // failed to setup the decoder
        error_code = 2;
        goto failure;
    }

    /* Read the main header of the codestream and if necessary the JP2 boxes*/
    if (! opj_read_header(stream, codec, &image))
    {
        // failed to read the header
        error_code = 3;
        goto failure;
    }

    // TODO: add check that all components match

    if (parameters.numcomps)
    {
        if (!opj_set_decoded_components(
                codec, parameters.numcomps,
                parameters.comps_indices, OPJ_FALSE)
            )
        {
            // failed to set the component indices
            error_code = 4;
            goto failure;
        }
    }

    if (!opj_set_decode_area(
            codec, image,
            (OPJ_INT32)parameters.DA_x0,
            (OPJ_INT32)parameters.DA_y0,
            (OPJ_INT32)parameters.DA_x1,
            (OPJ_INT32)parameters.DA_y1)
        )
    {
        // failed to set the decoded area
        error_code = 5;
        goto failure;
    }

    /* Get the decoded image */
    if (!(opj_decode(codec, stream, image) && opj_end_decompress(codec, stream)))
    {
        // failed to decode image
        error_code = 6;
        goto failure;
    }

    // Set our component pointers
    const unsigned int NR_COMPONENTS = image->numcomps;  // 15444-1 A.5.1
    p_component = malloc(NR_COMPONENTS * sizeof(int *));
    for (unsigned int ii = 0; ii < NR_COMPONENTS; ii++)
    {
        p_component[ii] = image->comps[ii].data;
    }

    int width = (int)image->comps[0].w;
    int height = (int)image->comps[0].h;
    int precision = (int)image->comps[0].prec;
    int mask = (1 << precision) - 1;

    // Our output should have planar configuration of 0, i.e. for RGB data
    //  we have R1, B1, G1 | R2, G2, B2 | ..., where 1 is the first pixel,
    //  2 the second, etc
    // See DICOM Standard, Part 3, Annex C.7.6.3.1.3
    if (precision <= 8)
    {
        // 8-bit signed/unsigned
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                for (unsigned int ii = 0; ii < NR_COMPONENTS; ii++)
                {
                    *out = (unsigned char)(*p_component[ii] & mask);
                    out++;
                    p_component[ii]++;
                }
            }
        }
    }
    else if (precision <= 16)
    {
        union {
            unsigned short val;
            unsigned char vals[2];
        } u16;

        // 16-bit signed/unsigned
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                for (unsigned int ii = 0; ii < NR_COMPONENTS; ii++)
                {
                    u16.val = (unsigned short)(*p_component[ii] & mask);
                    *out = u16.vals[0];
                    out++;
                    *out = u16.vals[1];
                    out++;
                    p_component[ii]++;
                }
            }
        }
    }
    else
    {
        // Support for more than 16-bits per component is not implemented
        error_code = 7;
        goto failure;
    }

    if (p_component)
    {
        free(p_component);
        p_component = NULL;
    }
    destroy_parameters(&parameters);
    opj_destroy_codec(codec);
    opj_image_destroy(image);
    opj_stream_destroy(stream);

    return EXIT_SUCCESS;

    failure:
        if (p_component)
        {
            free(p_component);
            p_component = NULL;
        }
        destroy_parameters(&parameters);
        if (codec)
            opj_destroy_codec(codec);
        if (image)
            opj_image_destroy(image);
        if (stream)
            opj_stream_destroy(stream);

        return error_code;
}
