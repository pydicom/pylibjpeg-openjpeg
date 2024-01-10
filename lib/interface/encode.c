

#include "Python.h"
#include "numpy/ndarrayobject.h"
#include <stdlib.h>
#include <stdio.h>
#include <../openjpeg/src/lib/openjp2/openjpeg.h>

// Size of the buffer for the input stream
#define BUFFER_SIZE OPJ_J2K_STREAM_CHUNK_SIZE


static void info_callback(const char *msg, void *client_data)
{
    (void)client_data;
    fprintf(stdout, "[INFO] %s", msg);
}


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


static OPJ_SIZE_T py_write(void *src, OPJ_SIZE_T nr_bytes, void *dst)
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


extern int Encode(
    PyArrayObject *arr,
    void *dst,
    int codec_format,
    int bits_stored,
    int photometric_interpretation,
    int lossless,
    int use_mct,
    int compression_ratio
)
{
    /* Encode a numpy ndarray using JPEG 2000.

    Parameters
    ----------
    arr : PyArrayObject *
        The numpy ndarray containing the image data to be encoded.
        TODO: can we get rows, columns, samples per pixel from it?
    dst : PyObject *
        The destination for the encoded codestream, should be a bytearray.
    out : unsigned char *
      The Python stream object to write the encoded data to.
      TODO: how do we know how big it needs to be?
    codec_format : int
        The format of the encoded JPEG 2000 data, one of:
        * ``0`` - OPJ_CODEC_J2K : JPEG-2000 codestream
        * ``2`` - OPJ_CODEC_JP2 : JP2 file format

    bits_stored : unsigned int
      Supported values: 1-16
    photometric_interpretation : unsigned int
      Supported values: 0-5
    lossless : unsigned int
      Supported values 0-1
    use_mct : unsigned int
      Supported values 0-1 (only if lossless 1), can't be used with subsampling
    compression_ratio : unsigned int
      Supported values 0-100? (only if lossless 0)
      hmm, specified per layer... list[int]?

    Returns
    -------
    int
        The exit status, 0 for success, failure otherwise.
    */
    // https://numpy.org/doc/stable/reference/c-api/array.html
    unsigned int return_code;

    // Check input
    // Determine the number of dimensions in the array, should be 2 or 3
    int nd = PyArray_NDIM(arr);

    // Can return NULL for 0-dimension arrays
    npy_intp *shape = PyArray_DIMS(arr);
    OPJ_UINT32 rows = 0;
    OPJ_UINT32 columns = 0;
    unsigned int samples_per_pixel = 0;
    switch (nd) {
      case 2: {
          // 1 sample per pixel
          samples_per_pixel = 1;
          rows = (OPJ_UINT32) shape[0];
          columns = (OPJ_UINT32) shape[1];
          break;
      }
      case 3: {
          // Only allow 3 or 4 samples per pixel
          if ( shape[2] != 3 && shape[2] != 4 ) {
            return 2;
          }
          rows = (OPJ_UINT32) shape[0];
          columns = (OPJ_UINT32) shape[1];
          samples_per_pixel = (unsigned int) shape[2];
          break;
      }
      default: {
        return 1;
      }
    }

    // Check number of rows and columns is in (1, 65535)
    if (rows < 1 || rows > 65535) {
        return 5;
    }
    if (columns < 1 || columns > 65535) {
        return 6;
    }

    // Check the dtype is supported
    PyArray_Descr *dtype = PyArray_DTYPE(arr);
    int type_enum = dtype->type_num;
    switch (type_enum) {
      case 0:  // bool
      case 1:  // i1
      case 2:  // u1
      case 3:  // i2
      case 4:  // u2
        break;
      default:
        return 3;
    }


    // Check data uses little endian byte order
    // TODO: perform byte-swapping instead?
    if (0) {
        // This is what PyArray_GetEndianness() is doing...
        const union {
            npy_uint32 i;
            char c[4];
        } bint = {0x01020304};

        if (bint.c[0] == 1) {
            printf("big endian\n");
        }
        else if (bint.c[0] == 4) {
            printf("little endian\n");
        }

        char byteorder = dtype->byteorder;
        // PyArray_GetEndianness segfaults for some reason
        int endianness = PyArray_GetEndianness();
        if (type_enum == 3 || type_enum == 4) {
            // <: little, >: big, =: native, |: irrelevant
            // single-quotes for char, double-quotes for strings
            if (byteorder == '=' && endianness == NPY_BIG) {
                return 4;
            } else if (byteorder == '>') {
                return 4;
            }
        }
    }

    // Check if array is C-style, contiguous and aligned
    if (PyArray_ISCARRAY_RO(arr) != 1) {
        return 7;
    };

    unsigned int bits_allocated;
    if (type_enum == 0 || type_enum == 1 || type_enum == 2) {
        // bool, i1, u1
        bits_allocated = 8;
    } else {
        // i2, u2
        bits_allocated = 16;
    }

    // Check precision in (1, bits_allocated)
    if (bits_stored < 1 || bits_stored > bits_allocated) {
        return 8;
    }

    // Check photometric_interpretation is supported and valid
    if (photometric_interpretation < 0 && photometric_interpretation > 5) {
        return 9;
    }

    // OPJ_CLRSPC_GRAY requires greyscale input
    if (photometric_interpretation == 1 && samples_per_pixel != 2) {
        return 10;
    }

    if (photometric_interpretation == 1 && samples_per_pixel != 3) {
        // OPJ_CLRSPC_SRGB
        return 11;
    } else if (photometric_interpretation == 3 && samples_per_pixel != 3) {
        // OPJ_CLRSPC_SYCC
        return 11;
    } else if (photometric_interpretation == 4 && samples_per_pixel != 3) {
      // OPJ_CLRSPC_EYCC
        return 11;
    } else if (photometric_interpretation == 5 && samples_per_pixel != 4) {
        // OPJ_CLRSPC_CMYK
        return 12;
    }

    // Check lossless/lossy value is valid
    if (lossless != 0 && lossless != 1) {
        return 13;
    }

    // Check MCT value is valid
    // TODO: disallow mct if not samples_per_pixel 3
    if (lossless == 0 && use_mct != 0) {
        return 14;
    } else if (lossless == 1 && (use_mct != 0 && use_mct != 1)) {
        return 15;
    }
    printf("  mct %d\n", use_mct);
    printf("  lossless %d\n", lossless);

    unsigned int is_signed;
    if (type_enum == 1 || type_enum == 3) {
      is_signed = 1;
    } else {
      is_signed = 0;
    }
    printf("  is signed %d\n", is_signed);

    // printf("Encoding start...\n");
    printf("Input validation complete\n");


    // Encoding parameters
    opj_cparameters_t parameters;
    opj_stream_t *stream = 00;
    opj_codec_t *l_codec = 00;

    opj_set_default_encoder_parameters(&parameters);

    // Set lossy (if applicable)
    // if (lossless == 0) {
    //   parameters->irreversible = 1;
    //   // TODO: set rates for each layer?
    // }

    // Set MCT
    parameters.tcp_mct = use_mct;

    // Set codec
    parameters.cod_format = codec_format;
    printf("  cod_format %d\n", parameters.cod_format);

    // lossless setup - not needed post v2.2.0
    if (parameters.tcp_numlayers == 0) {
        parameters.tcp_rates[0] = 0;
        parameters.tcp_numlayers++;
        parameters.cp_disto_alloc = 1;
    }

    // Input image
    printf("Creating the opj_image_t...\n");
    opj_image_t *image = NULL;

    // Setup the parameters for each image component
    opj_image_cmptparm_t *cmptparm;
    cmptparm = (opj_image_cmptparm_t*) calloc(
        (OPJ_UINT32) samples_per_pixel,
        sizeof(opj_image_cmptparm_t)
    );
    if (!cmptparm) {
        opj_stream_destroy(stream);
        opj_destroy_codec(l_codec);
        opj_image_destroy(image);
        return 17;
    }
    unsigned int i;
    for (i = 0; i < samples_per_pixel; i++) {
        cmptparm[i].prec = (OPJ_UINT32) bits_stored;
        cmptparm[i].sgnd = (OPJ_UINT32) is_signed;
        // Sub-sampling: none
        cmptparm[i].dx = (OPJ_UINT32) 1;
        cmptparm[i].dy = (OPJ_UINT32) 1;
        cmptparm[i].w = columns;
        cmptparm[i].h = rows;
        printf("  prec %d, sgnd %d, dx %d, dy %d, w %d, h %d\n",
          cmptparm[i].prec,
          cmptparm[i].sgnd,
          cmptparm[i].dx,
          cmptparm[i].dx,
          cmptparm[i].w,
          cmptparm[i].h
        );
    }

    image = opj_image_create(
        (OPJ_UINT32) samples_per_pixel,
        &cmptparm[0],
        // photometric_interpretation
        OPJ_CLRSPC_GRAY
    );

    free(cmptparm);
    if (!image) {
      opj_stream_destroy(stream);
      opj_destroy_codec(l_codec);
      opj_image_destroy(image);
      return 18;
    }

    /* set image offset and reference grid */
    image->x0 = (OPJ_UINT32)parameters.image_offset_x0;
    image->y0 = (OPJ_UINT32)parameters.image_offset_y0;
    image->x1 = (OPJ_UINT32)parameters.image_offset_x0 + (OPJ_UINT32)(columns - 1) + 1;
    image->y1 = (OPJ_UINT32)parameters.image_offset_y0 + (OPJ_UINT32)(rows - 1) + 1;

    // Add the image data
    printf("Adding image data...\n");
    void *ptr;
    unsigned int p, r, c;
    if (bits_allocated == 8) {
      // bool, u1, i1
      // Planes
      for (p = 0; p < samples_per_pixel; p++)
      {
          // Rows
          for (r = 0; r < rows; r++)
          {
              // Columns
              for (c = 0; c < columns; c++)
              {
                  ptr = PyArray_GETPTR3(arr, r, c, p);
                  image->comps[p].data[c + columns * r] = is_signed ? *(npy_int8 *) ptr : *(npy_uint8 *) ptr;
              }
          }
      }
    } else {
        // u2, i2
        // Planes
        for (p = 0; p < samples_per_pixel; p++)
        {
            // Rows
            for (r = 0; r < rows; r++)
            {
                // Columns
                for (c = 0; c < columns; c++)
                {
                    ptr = PyArray_GETPTR3(arr, r, c, p);
                    image->comps[p].data[c + columns * r] = is_signed ? *(npy_int16 *) ptr : *(npy_uint16 *) ptr;
                }
            }
        }
    }

    printf("Image data has been added from the ndarray\n");

    /* encode the destination image */
    /* ---------------------------- */

    /* Get an encoder handle */
    switch (parameters.cod_format) {
        case 0: { /* JPEG-2000 codestream */
            l_codec = opj_create_compress(OPJ_CODEC_J2K);
            break;
        }
        case 2: { /* JPEG 2000 codestream + JPIP */
            l_codec = opj_create_compress(OPJ_CODEC_JP2);
            break;
        }
        default:
          opj_stream_destroy(stream);
          opj_destroy_codec(l_codec);
          opj_image_destroy(image);
          return 16;
    }
    printf("  Encoder handle gotten\n");

    /* catch events using our callbacks and give a local context */
    opj_set_info_handler(l_codec, info_callback, 00);
    opj_set_warning_handler(l_codec, info_callback, 00);
    opj_set_error_handler(l_codec, info_callback, 00);
    printf("  Callbacks set\n");

    if (! opj_setup_encoder(l_codec, &parameters, image)) {
        fprintf(stderr, "failed to encode image: opj_setup_encoder\n");
        opj_destroy_codec(l_codec);
        opj_image_destroy(image);
        opj_stream_destroy(stream);
        return 17;
    }

    printf("  Encoder setup complete");

    // TODO: where to write to?
    // Creates an abstract output stream; allocates memory
    stream = opj_stream_create(BUFFER_SIZE, OPJ_FALSE);

    if (!stream)
    {
        // Failed to create the input stream
        opj_stream_destroy(stream);
        opj_destroy_codec(l_codec);
        opj_image_destroy(image);
        return 18;
    }

    // Functions for the stream
    opj_stream_set_write_function(stream, py_write);
    opj_stream_set_skip_function(stream, py_skip);
    opj_stream_set_seek_function(stream, py_seek_set);
    opj_stream_set_user_data(stream, dst, NULL);

    printf("Encoding the image data...\n");
    OPJ_BOOL bSuccess;
    /* encode the image */
    bSuccess = opj_start_compress(l_codec, image, stream);
    if (!bSuccess)  {
        fprintf(stderr, "failed to encode image: opj_start_compress\n");
    }

    bSuccess = bSuccess && opj_encode(l_codec, stream);
    if (!bSuccess)  {
        fprintf(stderr, "failed to encode image: opj_encode\n");
    }

    bSuccess = bSuccess && opj_end_compress(l_codec, stream);
    if (!bSuccess)  {
        fprintf(stderr, "failed to encode image: opj_end_compress\n");
    }
    printf("Image data encoded?\n");

    opj_stream_destroy(stream);
    opj_destroy_codec(l_codec);
    opj_image_destroy(image);

    return 0;
}
//   // Customise parameters as required
//   // * Lossy/lossless
//   // * Image size, samples, bit depth,
//   // * Colourspace
//
//   // 0: JPEG2000 codestream only
//   // 2: JPEG200 codestream with JPIP
//   unsigned int codec = 0;  // OPJ_CODEC_FORMAT.OPJ_CODEC_J2K | OPJ_CODEC_JP2
//   // 1 to 65535
//   unsigned int rows, columns;
//   // 1, 3 or 4
//   unsigned int samples_per_pixel;
//   // 0: Unspecified
//   // 1: sRGB - only with samples_per_pixel 3
//   // 2: Grayscale - only with samples_per_pixel 1
//   // 3: sYCC (YCbCr) - only with samples_per_pixel 1
//   // 4: eYCC - ???
//   // 5: CMYK - only with samples_per_pixel 4
//   unsigned int photometric_interpretation;  // OPJ_COLOR_SPACE.OPJ_CLRSPC_SRGB | ...
//   // bit depth per pixel (precision)
//   // 1 to 16 -> push goal is test 24 and 32
//   unsigned int bits_stored;
//   // unsigned integer input
//   // signed integer input
//   unsigned int is_signed;  // bool?
//   // 0 for lossy
//   // 1 for lossless
//   unsigned int is_lossless;  // bool?
//
//   // TODO: Validate input
//
//   if is_lossless {
//     // Use reversible DWT 5-3
//     parameters.mode = 0;
//   } else {
//     // Use irreversible DWT 9-7
//     parameters.mode = 1;
//   }
//
//   // Use multiple component transformation if input is RGB
//   // TODO: how to tell if input is RGB?
//   // char tcp_mct
//   if samples_per_pixel == 3 and is_lossless and photometric_interpretation == "sRGB" {
//     // MCT on
//     parameters.tcp_mct = 1;
//   } else {
//     // MCT off
//     parameters.tcp_mct = 0;
//   }
//
//   cmptparm = (opj_image_cmptparm_t*) calloc((OPJ_UINT32)numcomps,
//              sizeof(opj_image_cmptparm_t));
//   if (!cmptparm) {
//       fprintf(stderr, "Failed to allocate image components parameters !!\n");
//       fprintf(stderr, "Aborting\n");
//       fclose(f);
//       return NULL;
//   }
//   /* initialize image components */
//   for (i = 0; i < numcomps; i++) {
//       cmptparm[i].prec = (OPJ_UINT32)raw_cp->rawBitDepth;
//       cmptparm[i].sgnd = (OPJ_UINT32)raw_cp->rawSigned;
//       cmptparm[i].dx = (OPJ_UINT32)(subsampling_dx * raw_cp->rawComps[i].dx);
//       cmptparm[i].dy = (OPJ_UINT32)(subsampling_dy * raw_cp->rawComps[i].dy);
//       cmptparm[i].w = (OPJ_UINT32)w;
//       cmptparm[i].h = (OPJ_UINT32)h;
//   }
//
//   /* create the image */
//   image = opj_image_create((OPJ_UINT32)numcomps, &cmptparm[0], color_space);
//   free(cmptparm);
//   if (!image) {
//       fclose(f);
//       return NULL;
//   }
//
//   /* set image offset and reference grid */
//   image->x0 = (OPJ_UINT32)parameters->image_offset_x0;
//   image->y0 = (OPJ_UINT32)parameters->image_offset_y0;
//   image->x1 = (OPJ_UINT32)parameters->image_offset_x0 + (OPJ_UINT32)(w - 1) *
//               (OPJ_UINT32)subsampling_dx + 1;
//   image->y1 = (OPJ_UINT32)parameters->image_offset_y0 + (OPJ_UINT32)(h - 1) *
//               (OPJ_UINT32)subsampling_dy + 1;
//
//   if (raw_cp->rawBitDepth <= 8) {
//       unsigned char value = 0;
//       for (compno = 0; compno < numcomps; compno++) {
//           int nloop = (w * h) / (raw_cp->rawComps[compno].dx *
//                                  raw_cp->rawComps[compno].dy);
//           for (i = 0; i < nloop; i++) {
//               if (!fread(&value, 1, 1, f)) {
//                   fprintf(stderr, "Error reading raw file. End of file probably reached.\n");
//                   opj_image_destroy(image);
//                   fclose(f);
//                   return NULL;
//               }
//               image->comps[compno].data[i] = raw_cp->rawSigned ? (char)value : value;
//           }
//       }
//   } else if (raw_cp->rawBitDepth <= 16) {
//       unsigned short value;
//       for (compno = 0; compno < numcomps; compno++) {
//           int nloop = (w * h) / (raw_cp->rawComps[compno].dx *
//                                  raw_cp->rawComps[compno].dy);
//           for (i = 0; i < nloop; i++) {
//               unsigned char temp1;
//               unsigned char temp2;
//               if (!fread(&temp1, 1, 1, f)) {
//                   fprintf(stderr, "Error reading raw file. End of file probably reached.\n");
//                   opj_image_destroy(image);
//                   fclose(f);
//                   return NULL;
//               }
//               if (!fread(&temp2, 1, 1, f)) {
//                   fprintf(stderr, "Error reading raw file. End of file probably reached.\n");
//                   opj_image_destroy(image);
//                   fclose(f);
//                   return NULL;
//               }
//               if (big_endian) {
//                   value = (unsigned short)((temp1 << 8) + temp2);
//               } else {
//                   value = (unsigned short)((temp2 << 8) + temp1);
//               }
//               image->comps[compno].data[i] = raw_cp->rawSigned ? (short)value : value;
//           }
//       }
//   } else {
//       fprintf(stderr,
//               "OpenJPEG cannot encode raw components with bit depth higher than 16 bits.\n");
//       opj_image_destroy(image);
//       fclose(f);
//       return NULL;
//   }
//
//   // Seems unlikely
//   if (!image) {
//       fprintf(stderr, "Unable to load file: got no image\n");
//       ret = 1;
//       goto fin;
//   }
//
//   /* encode the destination image */
//   /* ---------------------------- */
//
//   switch (parameters.cod_format) {
//   case J2K_CFMT: { /* JPEG-2000 codestream */
//       /* Get a decoder handle */
//       l_codec = opj_create_compress(OPJ_CODEC_J2K);
//       break;
//   }
//   case JP2_CFMT: { /* JPEG 2000 compressed image data */
//       /* Get a decoder handle */
//       l_codec = opj_create_compress(OPJ_CODEC_JP2);
//       break;
//   }
//   default:
//       fprintf(stderr, "skipping file..\n");
//       opj_stream_destroy(l_stream);
//       continue;
//   }
//
//   /* catch events using our callbacks and give a local context */
//   opj_set_info_handler(l_codec, info_callback, 00);
//   opj_set_warning_handler(l_codec, warning_callback, 00);
//   opj_set_error_handler(l_codec, error_callback, 00);
//

//   if (! opj_setup_encoder(l_codec, &parameters, image)) {
//       fprintf(stderr, "failed to encode image: opj_setup_encoder\n");
//       opj_destroy_codec(l_codec);
//       opj_image_destroy(image);
//       ret = 1;
//       goto fin;
//   }
//
//   if (num_threads >= 1 &&
//           !opj_codec_set_threads(l_codec, num_threads)) {
//       fprintf(stderr, "failed to set number of threads\n");
//       opj_destroy_codec(l_codec);
//       opj_image_destroy(image);
//       ret = 1;
//       goto fin;
//   }
//
//   /* open a byte stream for writing and allocate memory for all tiles */
//   l_stream = opj_stream_create_default_file_stream(parameters.outfile, OPJ_FALSE);
//   if (! l_stream) {
//       ret = 1;
//       goto fin;
//   }
//
//   /* encode the image */
//   bSuccess = opj_start_compress(l_codec, image, l_stream);
//   if (!bSuccess)  {
//       fprintf(stderr, "failed to encode image: opj_start_compress\n");
//   }
//   if (bSuccess && bUseTiles) {
//       OPJ_BYTE *l_data;
//       OPJ_UINT32 l_data_size = 512 * 512 * 3;
//       l_data = (OPJ_BYTE*) calloc(1, l_data_size);
//       if (l_data == NULL) {
//           ret = 1;
//           goto fin;
//       }
//       for (i = 0; i < l_nb_tiles; ++i) {
//           if (! opj_write_tile(l_codec, i, l_data, l_data_size, l_stream)) {
//               fprintf(stderr, "ERROR -> test_tile_encoder: failed to write the tile %u!\n",
//                       i);
//               opj_stream_destroy(l_stream);
//               opj_destroy_codec(l_codec);
//               opj_image_destroy(image);
//               ret = 1;
//               goto fin;
//           }
//       }
//       free(l_data);
//   } else {
//       bSuccess = bSuccess && opj_encode(l_codec, l_stream);
//       if (!bSuccess)  {
//           fprintf(stderr, "failed to encode image: opj_encode\n");
//       }
//   }
//   bSuccess = bSuccess && opj_end_compress(l_codec, l_stream);
//   if (!bSuccess)  {
//       fprintf(stderr, "failed to encode image: opj_end_compress\n");
//   }
//
//   if (!bSuccess)  {
//       opj_stream_destroy(l_stream);
//       opj_destroy_codec(l_codec);
//       opj_image_destroy(image);
//       fprintf(stderr, "failed to encode image\n");
//       remove(parameters.outfile);
//       ret = 1;
//       goto fin;
//   }
//
//   /* close and free the byte stream */
//   opj_stream_destroy(l_stream);
//
//   /* free remaining compression structures */
//   opj_destroy_codec(l_codec);
//
//   /* free image data */
//   opj_image_destroy(image);
//
//   }
//
// fin:
//     if (parameters.cp_comment) {
//         free(parameters.cp_comment);
//     }
//     if (parameters.cp_matrice) {
//         free(parameters.cp_matrice);
//     }
//     if (raw_cp.rawComps) {
//         free(raw_cp.rawComps);
//     }
//     if (img_fol.imgdirpath) {
//         free(img_fol.imgdirpath);
//     }
//     if (dirptr) {
//         if (dirptr->filename_buf) {
//             free(dirptr->filename_buf);
//         }
//         if (dirptr->filename) {
//             free(dirptr->filename);
//         }
//         free(dirptr);
//     }
//     return ret;
// }
