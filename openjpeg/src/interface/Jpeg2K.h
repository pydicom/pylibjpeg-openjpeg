/*
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
*/

/*
 * The Python Imaging Library
 * $Id$
 *
 * declarations for the OpenJPEG codec interface.
 *
 * Copyright (c) 2014 by Coriolis Systems Limited
 * Copyright (c) 2014 by Alastair Houghton
 */

#include <openjpeg.h>

/* 1MB for now */
#define BUFFER_SIZE OPJ_J2K_STREAM_CHUNK_SIZE

/* -------------------------------------------------------------------- */
/* Decoder								*/
/* -------------------------------------------------------------------- */

typedef struct {
    /* CONFIGURATION */

    /* File descriptor, if available; otherwise, -1 */
    int fd;

    /* File pointer, when opened */
    FILE * pfile;

    /* Length of data, if available; otherwise, -1 */
    off_t length;

    /* Specify the desired format */
    OPJ_CODEC_FORMAT format;

    /* Set to divide image resolution by 2**reduce. */
    int            reduce;

    /* Set to limit the number of quality layers to decode (0 = all layers) */
    int            layers;

    /* PRIVATE CONTEXT (set by decoder) */
    const char    *error_msg;

} JPEG2KDECODESTATE;

/* -------------------------------------------------------------------- */
/* Encoder								*/
/* -------------------------------------------------------------------- */

typedef struct {
    /* CONFIGURATION */

    /* File descriptor, if available; otherwise, -1 */
    int           fd;

    /* File pointer, when opened */
    FILE * pfile;

    /* Specify the desired format */
    OPJ_CODEC_FORMAT format;

    /* Image offset */
    int            offset_x, offset_y;

    /* Tile information */
    int            tile_offset_x, tile_offset_y;
    int            tile_size_x, tile_size_y;

    /* Quality layers (a sequence of numbers giving *either* rates or dB) */
    int            quality_is_in_db;
    PyObject      *quality_layers;

    /* Number of resolutions (DWT decompositions + 1 */
    int            num_resolutions;

    /* Code block size */
    int            cblk_width, cblk_height;

    /* Precinct size */
    int            precinct_width, precinct_height;

    /* Compression style */
    int            irreversible;

    /* Progression order (LRCP/RLCP/RPCL/PCRL/CPRL) */
    OPJ_PROG_ORDER progression;

    /* Cinema mode */
    OPJ_CINEMA_MODE cinema_mode;

    /* PRIVATE CONTEXT (set by decoder) */
    const char    *error_msg;


} JPEG2KENCODESTATE;
