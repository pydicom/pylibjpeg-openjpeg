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
 * The Python Imaging Library.
 * $Id$
 *
 * decoder for JPEG2000 image data.
 *
 * history:
 * 2014-03-12 ajh  Created
 *
 * Copyright (c) 2014 Coriolis Systems Limited
 * Copyright (c) 2014 Alastair Houghton
 *
 * See the README file for details on usage and redistribution.
 */

//#include "Imaging.h"

//#ifdef HAVE_OPENJPEG

#include <stdlib.h>
#include "Jpeg2K.h"

typedef struct {
    OPJ_UINT32 tile_index;
    OPJ_UINT32 data_size;
    OPJ_INT32  x0, y0, x1, y1;
    OPJ_UINT32 nb_comps;
} JPEG2KTILEINFO;

/* -------------------------------------------------------------------- */
/* Error handler                                                        */
/* -------------------------------------------------------------------- */

static void j2k_error(const char *msg, void *client_data)
{
    JPEG2KDECODESTATE *state = (JPEG2KDECODESTATE *) client_data;
    free((void *)state->error_msg);
    state->error_msg = strdup(msg);
}

/* -------------------------------------------------------------------- */
/* Buffer input stream                                                  */
/* -------------------------------------------------------------------- */

static OPJ_SIZE_T j2k_read(
        void *p_buffer, OPJ_SIZE_T p_nb_bytes, void *p_user_data
)
{
    ImagingCodecState state = (ImagingCodecState)p_user_data;

    // FIXME: need to implement read for numpy char array ?
    // state->fd should go to state->stream or something
    size_t len = _imaging_read_pyFd(state->fd, p_buffer, p_nb_bytes);

    return len ? len : (OPJ_SIZE_T) - 1;
}


static OPJ_OFF_T j2k_skip(OPJ_OFF_T p_nb_bytes, void *p_user_data)
{
    off_t pos;
    ImagingCodecState state = (ImagingCodecState)p_user_data;

    // FIXME: need to implement seek and tell for numpy char array
    _imaging_seek_pyFd(state->fd, p_nb_bytes, SEEK_CUR);
    pos = _imaging_tell_pyFd(state->fd);

    return pos ? pos : (OPJ_OFF_T) - 1;
}

/* -------------------------------------------------------------------- */
/* Unpackers                                                            */
/* -------------------------------------------------------------------- */

// FIXME: Imaging here
typedef void (*j2k_unpacker_t)(
    opj_image_t *in, const JPEG2KTILEINFO *tileInfo,
    const UINT8 *data, Imaging im
);


// You know, I don't think these unpacker functions do anything but convert
//  the raw decoded data to the "correct" pillow image mode format
// Which is good, because it means they aren't needed at all
// Unsigned monochrome 8 bit output
// FIXME: Imaging here
static void j2ku_gray_l(
    opj_image_t *in, const JPEG2KTILEINFO *tileinfo, const UINT8 *tiledata,
    Imaging im)
{
    // x0, y0:
    unsigned x0 = tileinfo->x0 - in->x0, y0 = tileinfo->y0 - in->y0;
    // w:
    unsigned w = tileinfo->x1 - tileinfo->x0;
    // h:
    unsigned h = tileinfo->y1 - tileinfo->y0;

    // shift:
    int shift = 8 - in->comps[0].prec;
    // offset:
    int offset = in->comps[0].sgnd ? 1 << (in->comps[0].prec - 1) : 0;
    // csiz:
    int csiz = (in->comps[0].prec + 7) >> 3;

    unsigned x, y;

    if (csiz == 3)
        csiz = 4;

    if (shift < 0)
        offset += 1 << (-shift - 1);

    // csiz is number of components? or precision...? bytes?
    switch (csiz) {
    case 1:
        // for each row
        for (y = 0; y < h; ++y) {
            // `data` must be the pixel data for the current row
            const UINT8 *data = &tiledata[y * w];
            // pointer to corresponding row in the pillow image?
            UINT8 *row = (UINT8 *)im->image[y0 + y] + x0;
            // For each pixel in the row
            for (x = 0; x < w; ++x)
                // shift the raw pixel data, increment the row pointer
                *row++ = j2ku_shift(offset + *data++, shift);
        }
        break;
    case 2:
        for (y = 0; y < h; ++y) {
            const UINT16 *data = (const UINT16 *)&tiledata[2 * y * w];
            UINT8 *row = (UINT8 *)im->image[y0 + y] + x0;
            for (x = 0; x < w; ++x)
                *row++ = j2ku_shift(offset + *data++, shift);
        }
        break;
    case 4:
        for (y = 0; y < h; ++y) {
            const UINT32 *data = (const UINT32 *)&tiledata[4 * y * w];
            UINT8 *row = (UINT8 *)im->image[y0 + y] + x0;
            for (x = 0; x < w; ++x)
                *row++ = j2ku_shift(offset + *data++, shift);
        }
        break;
    }
}


// Unsigned monochrome 16 bit output
// FIXME: Imaging here
static void j2ku_gray_i(opj_image_t *in, const JPEG2KTILEINFO *tileinfo,
    const UINT8 *tiledata, Imaging im)
{
    unsigned x0 = tileinfo->x0 - in->x0, y0 = tileinfo->y0 - in->y0;
    unsigned w = tileinfo->x1 - tileinfo->x0;
    unsigned h = tileinfo->y1 - tileinfo->y0;

    int shift = 16 - in->comps[0].prec;
    int offset = in->comps[0].sgnd ? 1 << (in->comps[0].prec - 1) : 0;
    int csiz = (in->comps[0].prec + 7) >> 3;

    unsigned x, y;

    if (csiz == 3)
        csiz = 4;

    if (shift < 0)
        offset += 1 << (-shift - 1);

    switch (csiz) {
    case 1:
        for (y = 0; y < h; ++y) {
            const UINT8 *data = &tiledata[y * w];
            UINT16 *row = (UINT16 *)im->image[y0 + y] + x0;
            for (x = 0; x < w; ++x)
                *row++ = j2ku_shift(offset + *data++, shift);
        }
        break;
    case 2:
        for (y = 0; y < h; ++y) {
            const UINT16 *data = (const UINT16 *)&tiledata[2 * y * w];
            UINT16 *row = (UINT16 *)im->image[y0 + y] + x0;
            for (x = 0; x < w; ++x)
                *row++ = j2ku_shift(offset + *data++, shift);
        }
        break;
    case 4:
        for (y = 0; y < h; ++y) {
            const UINT32 *data = (const UINT32 *)&tiledata[4 * y * w];
            UINT16 *row = (UINT16 *)im->image[y0 + y] + x0;
            for (x = 0; x < w; ++x)
                *row++ = j2ku_shift(offset + *data++, shift);
        }
        break;
    }
}


static void j2ku_srgb_rgb(opj_image_t *in, const JPEG2KTILEINFO *tileinfo,
    const UINT8 *tiledata, Imaging im)
{
    unsigned x0 = tileinfo->x0 - in->x0, y0 = tileinfo->y0 - in->y0;
    unsigned w = tileinfo->x1 - tileinfo->x0;
    unsigned h = tileinfo->y1 - tileinfo->y0;

    int shifts[3], offsets[3], csiz[3];
    const UINT8 *cdata[3];
    const UINT8 *cptr = tiledata;
    unsigned n, x, y;

    for (n = 0; n < 3; ++n) {
        cdata[n] = cptr;
        shifts[n] = 8 - in->comps[n].prec;
        offsets[n] = in->comps[n].sgnd ? 1 << (in->comps[n].prec - 1) : 0;
        csiz[n] = (in->comps[n].prec + 7) >> 3;

        if (csiz[n] == 3)
            csiz[n] = 4;

        if (shifts[n] < 0)
            offsets[n] += 1 << (-shifts[n] - 1);

        cptr += csiz[n] * w * h;
    }

    for (y = 0; y < h; ++y) {
        const UINT8 *data[3];
        UINT8 *row = (UINT8 *)im->image[y0 + y] + x0 * 4;
        for (n = 0; n < 3; ++n)
            data[n] = &cdata[n][csiz[n] * y * w];

        for (x = 0; x < w; ++x) {
            for (n = 0; n < 3; ++n) {
                UINT32 word = 0;

                switch (csiz[n]) {
                case 1: word = *data[n]++; break;
                case 2: word = *(const UINT16 *)data[n]; data[n] += 2; break;
                case 4: word = *(const UINT32 *)data[n]; data[n] += 4; break;
                }

                row[n] = j2ku_shift(offsets[n] + word, shifts[n]);
            }
            row[3] = 0xff;
            row += 4;
        }
    }
}

static void j2ku_sycc_rgb(opj_image_t *in, const JPEG2KTILEINFO *tileinfo,
    const UINT8 *tiledata, Imaging im)
{
    unsigned x0 = tileinfo->x0 - in->x0, y0 = tileinfo->y0 - in->y0;
    unsigned w = tileinfo->x1 - tileinfo->x0;
    unsigned h = tileinfo->y1 - tileinfo->y0;

    int shifts[3], offsets[3], csiz[3];
    const UINT8 *cdata[3];
    const UINT8 *cptr = tiledata;
    unsigned n, x, y;

    for (n = 0; n < 3; ++n) {
        cdata[n] = cptr;
        shifts[n] = 8 - in->comps[n].prec;
        offsets[n] = in->comps[n].sgnd ? 1 << (in->comps[n].prec - 1) : 0;
        csiz[n] = (in->comps[n].prec + 7) >> 3;

        if (csiz[n] == 3)
            csiz[n] = 4;

        if (shifts[n] < 0)
            offsets[n] += 1 << (-shifts[n] - 1);

        cptr += csiz[n] * w * h;
    }

    for (y = 0; y < h; ++y) {
        const UINT8 *data[3];
        UINT8 *row = (UINT8 *)im->image[y0 + y] + x0 * 4;
        UINT8 *row_start = row;
        for (n = 0; n < 3; ++n)
            data[n] = &cdata[n][csiz[n] * y * w];

        for (x = 0; x < w; ++x) {
            for (n = 0; n < 3; ++n) {
                UINT32 word = 0;

                switch (csiz[n]) {
                case 1: word = *data[n]++; break;
                case 2: word = *(const UINT16 *)data[n]; data[n] += 2; break;
                case 4: word = *(const UINT32 *)data[n]; data[n] += 4; break;
                }

                row[n] = j2ku_shift(offsets[n] + word, shifts[n]);
            }
            row[3] = 0xff;
            row += 4;
        }

        ImagingConvertYCbCr2RGB(row_start, row_start, w);
    }
}

static void j2ku_srgba_rgba(opj_image_t *in, const JPEG2KTILEINFO *tileinfo,
    const UINT8 *tiledata, Imaging im)
{
    unsigned x0 = tileinfo->x0 - in->x0, y0 = tileinfo->y0 - in->y0;
    unsigned w = tileinfo->x1 - tileinfo->x0;
    unsigned h = tileinfo->y1 - tileinfo->y0;

    int shifts[4], offsets[4], csiz[4];
    const UINT8 *cdata[4];
    const UINT8 *cptr = tiledata;
    unsigned n, x, y;

    for (n = 0; n < 4; ++n) {
        cdata[n] = cptr;
        shifts[n] = 8 - in->comps[n].prec;
        offsets[n] = in->comps[n].sgnd ? 1 << (in->comps[n].prec - 1) : 0;
        csiz[n] = (in->comps[n].prec + 7) >> 3;

        if (csiz[n] == 3)
            csiz[n] = 4;

        if (shifts[n] < 0)
            offsets[n] += 1 << (-shifts[n] - 1);

        cptr += csiz[n] * w * h;
    }

    for (y = 0; y < h; ++y) {
        const UINT8 *data[4];
        UINT8 *row = (UINT8 *)im->image[y0 + y] + x0 * 4;
        for (n = 0; n < 4; ++n)
            data[n] = &cdata[n][csiz[n] * y * w];

        for (x = 0; x < w; ++x) {
            for (n = 0; n < 4; ++n) {
                UINT32 word = 0;

                switch (csiz[n]) {
                case 1: word = *data[n]++; break;
                case 2: word = *(const UINT16 *)data[n]; data[n] += 2; break;
                case 4: word = *(const UINT32 *)data[n]; data[n] += 4; break;
                }

                row[n] = j2ku_shift(offsets[n] + word, shifts[n]);
            }
            row += 4;
        }
    }
}

static void j2ku_sycca_rgba(opj_image_t *in, const JPEG2KTILEINFO *tileinfo,
    const UINT8 *tiledata, Imaging im)
{
    unsigned x0 = tileinfo->x0 - in->x0, y0 = tileinfo->y0 - in->y0;
    unsigned w = tileinfo->x1 - tileinfo->x0;
    unsigned h = tileinfo->y1 - tileinfo->y0;

    int shifts[4], offsets[4], csiz[4];
    const UINT8 *cdata[4];
    const UINT8 *cptr = tiledata;
    unsigned n, x, y;

    for (n = 0; n < 4; ++n) {
        cdata[n] = cptr;
        shifts[n] = 8 - in->comps[n].prec;
        offsets[n] = in->comps[n].sgnd ? 1 << (in->comps[n].prec - 1) : 0;
        csiz[n] = (in->comps[n].prec + 7) >> 3;

        if (csiz[n] == 3)
            csiz[n] = 4;

        if (shifts[n] < 0)
            offsets[n] += 1 << (-shifts[n] - 1);

        cptr += csiz[n] * w * h;
    }

    for (y = 0; y < h; ++y) {
        const UINT8 *data[4];
        UINT8 *row = (UINT8 *)im->image[y0 + y] + x0 * 4;
        UINT8 *row_start = row;
        for (n = 0; n < 4; ++n)
            data[n] = &cdata[n][csiz[n] * y * w];

        for (x = 0; x < w; ++x) {
            for (n = 0; n < 4; ++n) {
                UINT32 word = 0;

                switch (csiz[n]) {
                case 1: word = *data[n]++; break;
                case 2: word = *(const UINT16 *)data[n]; data[n] += 2; break;
                case 4: word = *(const UINT32 *)data[n]; data[n] += 4; break;
                }

                row[n] = j2ku_shift(offsets[n] + word, shifts[n]);
            }
            row += 4;
        }

        ImagingConvertYCbCr2RGB(row_start, row_start, w);
    }
}



/* -------------------------------------------------------------------- */
/* Decoder                                                              */
/* -------------------------------------------------------------------- */

enum {
    J2K_STATE_START = 0,
    J2K_STATE_DECODING = 1,
    J2K_STATE_DONE = 2,
    J2K_STATE_FAILED = 3,
};

// Decodes
static int j2k_decode_entry(Imaging im, ImagingCodecState state)
{
    JPEG2KDECODESTATE *context = (JPEG2KDECODESTATE *) state->context;
    // J2K stream
    opj_stream_t *stream = NULL;
    // struct defining image data and characteristics
    opj_image_t *image = NULL;
    // J2K codec
    opj_codec_t *codec = NULL;
    // struct with decompression parameters
    opj_dparameters_t params;
    // buffer size I guess
    size_t buffer_size = 0;
    unsigned n;

    // Creates an abstract input stream; allocates memory
    stream = opj_stream_create(BUFFER_SIZE, OPJ_TRUE);

    if (!stream) {
        state->errcode = IMAGING_CODEC_BROKEN;
        state->state = J2K_STATE_FAILED;
        goto quick_exit;
    }

    // j2k_read and j2k_skip are functions, but do we need this?
    opj_stream_set_read_function(stream, j2k_read);
    opj_stream_set_skip_function(stream, j2k_skip);

    // Sets the given data to be used as user data for the stream
    //  stream: the stream to modify
    //  state: the data to set
    //  NULL: function to free the data when obj_stream_destroy() is called
    opj_stream_set_user_data(stream, state, NULL);

    /* Hack: if we don't know the length, the largest file we can
       possibly support is 4GB.  We can't go larger than this, because
       OpenJPEG truncates this value for the final box in the file, and
       the box lengths in OpenJPEG are currently 32 bit. */
    if (context->length < 0)
        opj_stream_set_user_data_length(stream, 0xffffffff);
    else
        opj_stream_set_user_data_length(stream, context->length);

    /* Setup decompression context */
    context->error_msg = NULL;

    opj_set_default_decoder_parameters(&params);
    params.cp_reduce = context->reduce;
    params.cp_layer = context->layers;

    codec = opj_create_decompress(context->format);

    if (!codec) {
        state->errcode = IMAGING_CODEC_BROKEN;
        state->state = J2K_STATE_FAILED;
        goto quick_exit;
    }

    opj_set_error_handler(codec, j2k_error, context);
    opj_setup_decoder(codec, &params);

    // Decodes an image header
    //  stream: the j2k stream
    //  codec: the j2k codec to read
    //  image: the image structure with the characteristics of the encoded im
    if (!opj_read_header(stream, codec, &image)) {
        state->errcode = IMAGING_CODEC_BROKEN;
        state->state = J2K_STATE_FAILED;
        goto quick_exit;
    }

    /* Check that this image is something we can handle */
    if (image->numcomps < 1 || image->numcomps > 4
        || image->color_space == OPJ_CLRSPC_UNKNOWN) {
        state->errcode = IMAGING_CODEC_BROKEN;
        state->state = J2K_STATE_FAILED;
        goto quick_exit;
    }

    for (n = 1; n < image->numcomps; ++n) {
        if (image->comps[n].dx != 1 || image->comps[n].dy != 1) {
            state->errcode = IMAGING_CODEC_BROKEN;
            state->state = J2K_STATE_FAILED;
            goto quick_exit;
        }
    }

    /* Decode the image tile-by-tile; this means we only need use as much
       memory as is required for one tile's worth of components. */
    for (;;) {
        JPEG2KTILEINFO tile_info;
        OPJ_BOOL should_continue;
        unsigned correction = (1 << params.cp_reduce) - 1;

        // Reads a tile header, compulsory
        if (!opj_read_tile_header(
                codec,
                stream,
                &tile_info.tile_index,
                &tile_info.data_size,
                &tile_info.x0, &tile_info.y0,
                &tile_info.x1, &tile_info.y1,
                &tile_info.nb_comps,
                &should_continue
            )
        ){
            state->errcode = IMAGING_CODEC_BROKEN;
            state->state = J2K_STATE_FAILED;
            goto quick_exit;
        }

        if (!should_continue)
            break;

        /* Adjust the tile co-ordinates based on the reduction (OpenJPEG
           doesn't do this for us) */
        tile_info.x0 = (tile_info.x0 + correction) >> context->reduce;
        tile_info.y0 = (tile_info.y0 + correction) >> context->reduce;
        tile_info.x1 = (tile_info.x1 + correction) >> context->reduce;
        tile_info.y1 = (tile_info.y1 + correction) >> context->reduce;

        if (buffer_size < tile_info.data_size) {
            /* malloc check ok, tile_info.data_size from openjpeg */
            UINT8 *new = realloc (state->buffer, tile_info.data_size);
            if (!new) {
                state->errcode = IMAGING_CODEC_MEMORY;
                state->state = J2K_STATE_FAILED;
                goto quick_exit;
            }
            state->buffer = new;
            buffer_size = tile_info.data_size;
        }

        // Reads and decodes a tile's data, compulsory, should be called after
        //  opj_read_tile_header
        if (!opj_decode_tile_data(codec,
                                  tile_info.tile_index,
                                  (OPJ_BYTE *)state->buffer,
                                  tile_info.data_size,
                                  stream)) {
            state->errcode = IMAGING_CODEC_BROKEN;
            state->state = J2K_STATE_FAILED;
            goto quick_exit;
        }

        // FIXME: im - uses the output image size I believe - replaceable
        /* Check the tile bounds; if the tile is outside the image area,
           or if it has a negative width or height (i.e. the coordinates are
           swapped), bail. */
        if (tile_info.x0 >= tile_info.x1
            || tile_info.y0 >= tile_info.y1
            || tile_info.x0 < image->x0
            || tile_info.y0 < image->y0
            || tile_info.x1 - image->x0 > im->xsize
            || tile_info.y1 - image->y0 > im->ysize) {
            state->errcode = IMAGING_CODEC_BROKEN;
            state->state = J2K_STATE_FAILED;
            goto quick_exit;
        }

        // FIXME: im - fed to unpacker function but not necessary
        // Replace with a function that just writes the results to our output
        // array or whatever
        //unpack(image, &tile_info, state->buffer, im);
    }

    // Read after the codestream if necessary (?)
    if (!opj_end_decompress(codec, stream)) {
        state->errcode = IMAGING_CODEC_BROKEN;
        state->state = J2K_STATE_FAILED;
        goto quick_exit;
    }

    state->state = J2K_STATE_DONE;
    state->errcode = IMAGING_CODEC_END;

    if (context->pfile) {
        if(fclose(context->pfile)){
            context->pfile = NULL;
        }
    }

 quick_exit:
    if (codec)
        opj_destroy_codec(codec);
    if (image)
        opj_image_destroy(image);
    if (stream)
        opj_stream_destroy(stream);

    return -1;
}

int ImagingJpeg2KDecode(
    Imaging im, ImagingCodecState state, UINT8* buf, Py_ssize_t bytes
)
{

    if (bytes){
        state->errcode = IMAGING_CODEC_BROKEN;
        state->state = J2K_STATE_FAILED;
        return -1;
    }

    if (state->state == J2K_STATE_DONE || state->state == J2K_STATE_FAILED)
        return -1;

    if (state->state == J2K_STATE_START) {
        state->state = J2K_STATE_DECODING;

        return j2k_decode_entry(im, state);
    }

    if (state->state == J2K_STATE_DECODING) {
        state->errcode = IMAGING_CODEC_BROKEN;
        state->state = J2K_STATE_FAILED;
        return -1;
    }
    return -1;
}

/* -------------------------------------------------------------------- */
/* Cleanup                                                              */
/* -------------------------------------------------------------------- */

int ImagingJpeg2KDecodeCleanup(ImagingCodecState state) {
    JPEG2KDECODESTATE *context = (JPEG2KDECODESTATE *)state->context;

    if (context->error_msg) {
        free ((void *)context->error_msg);
    }

    context->error_msg = NULL;

    return -1;
}

/*const char * ImagingJpeg2KVersion(void)
{
    return opj_version();
}*/
