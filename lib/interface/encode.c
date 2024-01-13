

#include "Python.h"
#include "numpy/ndarrayobject.h"
#include <stdlib.h>
#include <stdio.h>
#include <../openjpeg/src/lib/openjp2/openjpeg.h>
#include "utils.h"


static void py_debug(const char *msg) {
    py_log("openjpeg.encode", "DEBUG", msg);
}

static void py_error(const char *msg) {
    py_log("openjpeg.encode", "ERROR", msg);
}

static void info_callback(const char *msg, void *callback) {
    py_log("openjpeg.encode", "INFO", msg);
}

static void warning_callback(const char *msg, void *callback) {
    py_log("openjpeg.encode", "WARNING", msg);
}

static void error_callback(const char *msg, void *callback) {
    py_error(msg);
}


extern int Encode(
    PyArrayObject *arr,
    PyObject *dst,
    int bits_stored,
    int photometric_interpretation,
    int use_mct,
    PyObject *compression_ratios,
    PyObject *signal_noise_ratios,
    int codec_format
)
{
    /* Encode a numpy ndarray using JPEG 2000.

    Parameters
    ----------
    arr : PyArrayObject *
        The numpy ndarray containing the image data to be encoded.
    dst : PyObject *
        The destination for the encoded codestream, should be a BinaryIO.
    bits_stored : int
        Supported values: 1-16
    photometric_interpretation : int
        Supported values: 0-5
    use_mct : int
        Supported values 0-1, can't be used with subsampling
    compression_ratios : list[float]
        Encode lossy with the specified compression ratio for each layer. The
        ratio should be decreasing with increasing layer, and may be 1 to signal
        the use of lossless encoding for that layer.
    signal_noise_ratios : list[float]
        Encode lossy with the specified peak signal-to-noise ratio. The ratio
        should be increasing with increasing layer, but may be 0 in the final
        layer to signal the use of lossless encoding for that layer.
    codec_format : int
        The format of the encoded JPEG 2000 data, one of:
        * ``0`` - OPJ_CODEC_J2K : JPEG-2000 codestream
        * ``2`` - OPJ_CODEC_JP2 : JP2 file format

    Returns
    -------
    int
        The exit status, 0 for success, failure otherwise.
    */

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
            // (rows, columns)
            samples_per_pixel = 1;
            rows = (OPJ_UINT32) shape[0];
            columns = (OPJ_UINT32) shape[1];
            break;
        }
        case 3: {
            // (rows, columns, planes)
            // Only allow 3 or 4 samples per pixel
            if ( shape[2] != 3 && shape[2] != 4 ) {
                py_error(
                    "The input array has an unsupported number of samples per pixel"
                );
                return 1;
            }
            rows = (OPJ_UINT32) shape[0];
            columns = (OPJ_UINT32) shape[1];
            samples_per_pixel = (unsigned int) shape[2];
            break;
        }
        default: {
            py_error("An input array with the given dimensions is not supported");
            return 2;
        }
    }

    // Check number of rows and columns is in (1, 65535)
    if (rows < 1 || rows > 65535) {
        py_error("The input array has an unsupported number of rows");
        return 3;
    }
    if (columns < 1 || columns > 65535) {
        py_error("The input array has an unsupported number of columns");
        return 4;
    }

    // Check the dtype is supported
    PyArray_Descr *dtype = PyArray_DTYPE(arr);
    int type_enum = dtype->type_num;
    switch (type_enum) {
        case NPY_BOOL:  // bool
        case NPY_INT8:  // i1
        case NPY_UINT8:  // u1
        case NPY_INT16:  // i2
        case NPY_UINT16:  // u2
        case NPY_INT32:  // i4
        case NPY_UINT32:  // u4
            break;
        default:
            py_error("The input array has an unsupported dtype");
            return 5;
    }

    // Check array is C-style, contiguous and aligned
    if (PyArray_ISCARRAY_RO(arr) != 1) {
        py_error("The input array must be C-style, contiguous and aligned");
        return 7;
    };

    int bits_allocated;
    if (type_enum == NPY_BOOL || type_enum == NPY_INT8 || type_enum == NPY_UINT8) {
        bits_allocated = 8;  // bool, i1, u1
    } else if (type_enum == NPY_INT16 || type_enum == NPY_UINT16 )  {
        bits_allocated = 16;  // i2, u2
    } else {
        bits_allocated = 32;  // i4, u4
    }

    // Check `photometric_interpretation` is valid
    if (
        samples_per_pixel == 1
        && (
            photometric_interpretation != 0  // OPJ_CLRSPC_UNSPECIFIED
            && photometric_interpretation != 2  // OPJ_CLRSPC_GRAY
        )
    ) {
        py_error(
            "The value of the 'photometric_interpretation' parameter is not "
            "valid for the number of samples per pixel"
        );
        return 9;
    }

    if (
        samples_per_pixel == 3
        && (
            photometric_interpretation != 0  // OPJ_CLRSPC_UNSPECIFIED
            && photometric_interpretation != 1  // OPJ_CLRSPC_SRGB
            && photometric_interpretation != 3  // OPJ_CLRSPC_SYCC
            && photometric_interpretation != 4  // OPJ_CLRSPC_EYCC
        )
    ) {
        py_error(
            "The value of the 'photometric_interpretation' parameter is not "
            "valid for the number of samples per pixel"
        );
        return 9;
    }

    if (
        samples_per_pixel == 4
        && (
            photometric_interpretation != 0  // OPJ_CLRSPC_UNSPECIFIED
            && photometric_interpretation != 5  // OPJ_CLRSPC_CMYK
        )
    ) {
        py_error(
            "The value of the 'photometric_interpretation' parameter is not "
            "valid for the number of samples per pixel"
        );
        return 9;
    }

    // Disable MCT if the input is not RGB
    if (samples_per_pixel != 3 || photometric_interpretation != 1) {
        use_mct = 0;
    }

    // Check the encoding format
    if (codec_format != 0 && codec_format != 2) {
        py_error("The value of the 'codec_format' parameter is invalid");
        return 10;
    }

    unsigned int is_signed;
    if (type_enum == NPY_INT8 || type_enum == NPY_INT16 || type_enum == NPY_INT32) {
      is_signed = 1;
    } else {
      is_signed = 0;
    }

    // Encoding parameters
    unsigned int return_code;
    opj_cparameters_t parameters;
    opj_stream_t *stream = 00;
    opj_codec_t *codec = 00;
    opj_image_t *image = NULL;

    // subsampling_dx 1
    // subsampling_dy 1
    // tcp_numlayers = 0
    // tcp_rates[0] = 0
    // prog_order = OPJ_LRCP
    // cblockw_init = 64
    // cblockh_init = 64
    // numresolution = 6
    opj_set_default_encoder_parameters(&parameters);

    // Set MCT and codec
    parameters.tcp_mct = use_mct;
    parameters.cod_format = codec_format;

    // Set up for lossy (if applicable)
    Py_ssize_t nr_cr_layers = PyObject_Length(compression_ratios);
    Py_ssize_t nr_snr_layers = PyObject_Length(signal_noise_ratios);
    if (nr_cr_layers > 0 || nr_snr_layers > 0) {
        // Lossy compression using compression ratios
        parameters.irreversible = 1;  // use DWT 9-7
        if (nr_cr_layers > 0) {
            if (nr_cr_layers > 100) {
                return_code = 11;
                goto failure;
            }

            parameters.cp_disto_alloc = 1;  // Allocation by rate/distortion
            parameters.tcp_numlayers = nr_cr_layers;
            for (int idx = 0; idx < nr_cr_layers; idx++) {
                PyObject *item = PyList_GetItem(compression_ratios, idx);
                if (item == NULL || !PyFloat_Check(item)) {
                    return_code = 12;
                    goto failure;
                }
                double value = PyFloat_AsDouble(item);
                if (value < 1 || value > 1000) {
                    return_code = 13;
                    goto failure;
                }
                // Maximum 100 rates
                parameters.tcp_rates[idx] = value;
            }
            py_debug("Encoding using lossy compression based on compression ratios");

        } else {
            // Lossy compression using peak signal-to-noise ratios
            if (nr_snr_layers > 100) {
                return_code = 14;
                goto failure;
            }

            parameters.cp_fixed_quality = 1;
            parameters.tcp_numlayers = nr_snr_layers;
            for (int idx = 0; idx < nr_snr_layers; idx++) {
                PyObject *item = PyList_GetItem(signal_noise_ratios, idx);
                if (item == NULL || !PyFloat_Check(item)) {
                    return_code = 15;
                    goto failure;
                }
                double value = PyFloat_AsDouble(item);
                if (value < 0 || value > 1000) {
                    return_code = 16;
                    goto failure;
                }
                // Maximum 100 ratios
                parameters.tcp_distoratio[idx] = value;
            }
            py_debug(
                "Encoding using lossy compression based on peak signal-to-noise ratios"
            );
        }
    }

    py_debug("Input validation complete, setting up for encoding");

    // Create the input image and configure it
    // Setup the parameters for each image component
    opj_image_cmptparm_t *cmptparm;
    cmptparm = (opj_image_cmptparm_t*) calloc(
        (OPJ_UINT32) samples_per_pixel,
        sizeof(opj_image_cmptparm_t)
    );
    if (!cmptparm) {
        py_error("Failed to assign the image component parameters");
        return_code = 20;
        goto failure;
    }
    unsigned int i;
    for (i = 0; i < samples_per_pixel; i++) {
        cmptparm[i].prec = (OPJ_UINT32) bits_stored;
        cmptparm[i].sgnd = (OPJ_UINT32) is_signed;
        // Sub-sampling: none
        cmptparm[i].dx = 1;
        cmptparm[i].dy = 1;
        cmptparm[i].w = columns;
        cmptparm[i].h = rows;
    }

    // Create the input image object
    image = opj_image_create(
        (OPJ_UINT32) samples_per_pixel,
        &cmptparm[0],
        photometric_interpretation
    );

    free(cmptparm);
    if (!image) {
        py_error("Failed to create an empty image object");
        return_code = 21;
        goto failure;
    }

    /* set image offset and reference grid */
    image->x0 = (OPJ_UINT32)parameters.image_offset_x0;
    image->y0 = (OPJ_UINT32)parameters.image_offset_y0;
    image->x1 = (OPJ_UINT32)parameters.image_offset_x0 + (OPJ_UINT32)columns;
    image->y1 = (OPJ_UINT32)parameters.image_offset_y0 + (OPJ_UINT32)rows;

    // Add the image data
    void *ptr;
    unsigned int p, r, c;
    if (bits_allocated == 8) {  // bool, u1, i1
        for (p = 0; p < samples_per_pixel; p++)
        {
            for (r = 0; r < rows; r++)
            {
                for (c = 0; c < columns; c++)
                {
                    ptr = PyArray_GETPTR3(arr, r, c, p);
                    image->comps[p].data[c + columns * r] = is_signed ? *(npy_int8 *) ptr : *(npy_uint8 *) ptr;
                }
            }
        }
    } else if (bits_allocated == 16) {  // u2, i2
        for (p = 0; p < samples_per_pixel; p++)
        {
            for (r = 0; r < rows; r++)
            {
                for (c = 0; c < columns; c++)
                {
                    ptr = PyArray_GETPTR3(arr, r, c, p);
                    image->comps[p].data[c + columns * r] = is_signed ? *(npy_int16 *) ptr : *(npy_uint16 *) ptr;
                }
            }
        }
    } else if (bits_allocated == 32) {  // u4, i4
        for (p = 0; p < samples_per_pixel; p++)
        {
            for (r = 0; r < rows; r++)
            {
                for (c = 0; c < columns; c++)
                {
                    ptr = PyArray_GETPTR3(arr, r, c, p);
                    // `data[...]` is OPJ_INT32, so *may* support range i (1, 32), u (1, 31)
                    image->comps[p].data[c + columns * r] = is_signed ? *(npy_int32 *) ptr : *(npy_uint32 *) ptr;
                }
            }
        }
    }
    py_debug("Input image configured and populated with data");

    /* Get an encoder handle */
    switch (parameters.cod_format) {
        case 0: {  // J2K codestream only
            codec = opj_create_compress(OPJ_CODEC_J2K);
            break;
        }
        case 2: { // JP2 codestream
            codec = opj_create_compress(OPJ_CODEC_JP2);
            break;
        }
        default:
            py_error("Failed to set the encoding handler");
            return_code = 22;
            goto failure;
    }

    /* Send info, warning, error message to Python logging */
    opj_set_info_handler(codec, info_callback, NULL);
    opj_set_warning_handler(codec, warning_callback, NULL);
    opj_set_error_handler(codec, error_callback, NULL);

    if (! opj_setup_encoder(codec, &parameters, image)) {
        py_error("Failed to set up the encoder");
        return_code = 23;
        goto failure;
    }

    // Creates an abstract output stream; allocates memory
    stream = opj_stream_create(BUFFER_SIZE, OPJ_FALSE);

    if (!stream) {
        py_error("Failed to create the output stream");
        return_code = 24;
        goto failure;
    }

    // Functions for the stream
    opj_stream_set_write_function(stream, py_write);
    opj_stream_set_skip_function(stream, py_skip);
    opj_stream_set_seek_function(stream, py_seek_set);
    opj_stream_set_user_data(stream, dst, NULL);

    OPJ_BOOL result;

    // Encode `image` using `codec` and put the output in `stream`
    py_debug("Encoding started");
    result = opj_start_compress(codec, image, stream);
    if (!result)  {
        py_error("Failure result from 'opj_start_compress()'");
        return_code = 25;
        goto failure;
    }

    result = result && opj_encode(codec, stream);
    if (!result)  {
        py_error("Failure result from 'opj_encode()'");
        return_code = 26;
        goto failure;
    }

    result = result && opj_end_compress(codec, stream);
    if (!result)  {
        py_error("Failure result from 'opj_end_compress()'");
        return_code = 27;
        goto failure;
    }

    py_debug("Encoding completed");

    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    opj_image_destroy(image);

    return 0;

    failure:
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return return_code;
}
