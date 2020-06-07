
#include "Python.h"
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"


#define BUFFER_SIZE OPJ_J2K_STREAM_CHUNK_SIZE


const char * OpenJpegVersion(void)
{
    // return the openjpeg version as char array
    return opj_version();
}

typedef enum opj_prec_mode {
    OPJ_PREC_MODE_CLIP,
    OPJ_PREC_MODE_SCALE
} opj_precision_mode;

typedef struct opj_prec {
    OPJ_UINT32         prec;
    opj_precision_mode mode;
} opj_precision;

typedef struct opj_decompress_params {
    /** core library parameters */
    opj_dparameters_t core;

    /** input file name */
    char infile[OPJ_PATH_LEN];
    /** output file name */
    char outfile[OPJ_PATH_LEN];
    /** input file format 0: J2K, 1: JP2, 2: JPT */
    int decod_format;
    /** output file format 0: PGX, 1: PxM, 2: BMP */
    int cod_format;
    /** index file name */
    char indexfilename[OPJ_PATH_LEN];

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

    opj_precision* precision;
    OPJ_UINT32     nb_precision;

    /* force output colorspace to RGB */
    int force_rgb;
    /* upsample components according to their dx/dy values */
    int upsample;
    /* split output components to different files */
    int split_pnm;
    /** number of threads */
    int num_threads;
    /* Quiet */
    int quiet;
    /** number of components to decode */
    OPJ_UINT32 numcomps;
    /** indices of components to decode */
    OPJ_UINT32* comps_indices;
} opj_decompress_parameters;


enum {
    J2K_STATE_START = 0,
    J2K_STATE_DECODING = 1,
    J2K_STATE_DONE = 2,
    J2K_STATE_FAILED = 3,
    IMAGING_CODEC_BROKEN = -1,
};



typedef struct {
    OPJ_INT8 error_code;
    OPJ_INT8 state;
    //OPJ_UINT32 data_size;
    //OPJ_INT32  x0, y0, x1, y1;
    //OPJ_UINT32 nb_comps;
} CodecState;


Py_ssize_t tell_data(PyObject *fd)
{
    PyObject *result;
    Py_ssize_t location;

    printf("tell_data::Telling\n");

    result = PyObject_CallMethod(fd, "tell", NULL);
    location = PyLong_AsSsize_t(result);

    Py_DECREF(result);
    return location;
}


OPJ_SIZE_T read_data(PyObject* fd, char* destination, Py_ssize_t nr_bytes)
{
    PyObject* result;
    char* buffer;
    OPJ_SIZE_T length;
    int bytes_result;
    Py_ssize_t pos;

    printf("read_data::Reading %u bytes from input\n", nr_bytes);
    pos = tell_data(fd);
    printf("read_data::Start position: %d\n", pos);

    result = PyObject_CallMethod(fd, "read", "n", nr_bytes);
    bytes_result = PyBytes_AsStringAndSize(result, &buffer, &length);

    pos = tell_data(fd);

    //int val = <int>&buffer[0];

    printf("read_data::Read data of length: %u\n", length);
    printf("read_data::End position: %d\n", pos);

    if (bytes_result == -1) {
        goto error;
    }

    if (length > nr_bytes) {
        goto error;
    }
    //printf("Reading data: %d\n", buffer[0]);

    memcpy(destination, buffer, length);

    if (length > 6)
    {
        printf("read_data::First 6 bytes read are: ");
        for (int ii = 0; ii <= 5; ii++)
            printf("%x ", destination[ii] & 0xff);
        printf("\n");
    }

    //printf("Converting to size_t\n");
    //OPJ_SIZE_T newlen = (OPJ_SIZE_T)(PyLong_AsSize_t(length));
    //printf("Converted\n");

    Py_DECREF(result);
    return length;

error:
    printf("read_data::Error reading data");
    Py_DECREF(result);
    return -1;
}


int seek_data(PyObject *fd, OPJ_SIZE_T offset, int whence)
{
    PyObject *result;

    printf("seek_data::Seeking offset %d from %d\n", offset, whence);

    // n: convert C Py_ssize_t to a Python integer
    // i: convert C int to a Python integer
    result = PyObject_CallMethod(fd, "seek", "Ii", offset, whence);

    //printf("seek_data::current position: %d\n", (unsigned int)(result));

    Py_DECREF(result);
    return 0;
}



static OPJ_SIZE_T j2k_read(
    void *p_buffer, OPJ_SIZE_T p_nb_bytes, void *p_user_data
)
{
    //ImagingCodecState state = (ImagingCodecState)p_user_data;

    // FIXME: need to implement read for numpy char array ?
    // state->fd should go to state->stream or something
    size_t len = read_data(p_user_data, p_buffer, p_nb_bytes);

    return len ? len : (OPJ_SIZE_T) - 1;
}

static OPJ_BOOL j2k_seek(OPJ_OFF_T nr_bytes, void *p_user_data)
{
    /**
    * The function to be used as a seek function, the stream is
      then seekable, using SEEK_SET behavior.
    */
    // Python and C:
    //  SEEK_SET 0
    //  SEEK_CUR 1
    //  SEEK_END 2
    seek_data(p_user_data, nr_bytes, SEEK_SET);

    return OPJ_TRUE;
}

static OPJ_OFF_T j2k_skip(OPJ_OFF_T nr_bytes, void *p_user_data)
{
    off_t pos;

    seek_data(p_user_data, nr_bytes, SEEK_CUR);
    pos = tell_data(p_user_data);

    return pos ? pos : (OPJ_OFF_T)-1;
}

static OPJ_UINT64 get_input_size(void * p_user_data)
{
    OPJ_OFF_T input_length = 0;

    seek_data(p_user_data, 0, SEEK_END);
    input_length = (OPJ_OFF_T)tell_data(p_user_data);
    seek_data(p_user_data, 0, SEEK_SET);

    return (OPJ_UINT64)input_length;
}

/*static void j2k_error(const char *msg, void *client_data)
{
    //JPEG2KDECODESTATE *state = (JPEG2KDECODESTATE *) client_data;
    //free((void *)state->error_msg);
    //printf("%s", msg);
}*/


static void set_default_parameters(opj_decompress_parameters* parameters)
{
    if (parameters) {
        memset(parameters, 0, sizeof(opj_decompress_parameters));

        /* default decoding parameters (command line specific) */
        parameters->decod_format = -1;
        parameters->cod_format = -1;

        /* default decoding parameters (core) */
        opj_set_default_decoder_parameters(&(parameters->core));
    }
}


int decode(PyObject* fd, unsigned char *out) {
    // J2K stream
    opj_stream_t *stream = NULL;
    // struct defining image data and characteristics
    opj_image_t *image = NULL;
    // J2K codec
    opj_codec_t *codec = NULL;

    CodecState *state;

    opj_decompress_parameters parameters;
    set_default_parameters(&parameters);

    // Creates an abstract input stream; allocates memory
    stream = opj_stream_create(BUFFER_SIZE, OPJ_TRUE);

    if (!stream) {
        state->error_code = IMAGING_CODEC_BROKEN;
        state->state = J2K_STATE_FAILED;
        goto quick_exit;
    }

    opj_stream_set_read_function(stream, j2k_read);
    opj_stream_set_skip_function(stream, j2k_skip);
    opj_stream_set_seek_function(stream, j2k_seek);
    opj_stream_set_user_data(stream, fd, NULL);
    opj_stream_set_user_data_length(stream, get_input_size(fd));
    //opj_set_error_handler(codec, j2k_error, 00);

    // FIXME: should be automatic?
    codec = opj_create_decompress(OPJ_CODEC_J2K);  // JPEG-2000 codestream

    /* Setup the decoder decoding parameters using user parameters */
    if (!opj_setup_decoder(codec, &(parameters.core)))
    {
        fprintf(stderr, "ERROR -> opj_decompress: failed to setup the decoder\n");
        goto quick_exit;
    }

    if (parameters.num_threads >= 1 && !opj_codec_set_threads(codec, parameters.num_threads))
    {
        fprintf(stderr, "ERROR -> opj_decompress: failed to set number of threads\n");
        goto quick_exit;
    }

    /* Read the main header of the codestream and if necessary the JP2 boxes*/
    // Working! Excellent progress... :p
    if (! opj_read_header(stream, codec, &image)) {
        fprintf(stderr, "ERROR -> opj_decompress: failed to read the header\n");
        goto quick_exit;
    }

    printf("Number of components: %d\n", image->numcomps);
    printf("Colour space: %d\n", image->color_space);
    printf("Size: %d x %d\n", image->x1, image->y1);
    printf("Number of tiles to decode: %d\n", parameters.nb_tile_to_decode);

    if (parameters.numcomps) {
        if (!opj_set_decoded_components(
                codec, parameters.numcomps, parameters.comps_indices, OPJ_FALSE
            ))
        {
            fprintf(stderr, "ERROR -> opj_decompress: failed to set the component indices!\n");
            goto quick_exit;
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
        fprintf(stderr, "ERROR -> opj_decompress: failed to set the decoded area\n");
        goto quick_exit;
    }

    /* Get the decoded image */
    if (!(opj_decode(codec, stream, image) && opj_end_decompress(codec, stream)))
    {
        fprintf(stderr, "ERROR -> opj_decompress: failed to decode image!\n");
        goto quick_exit;
    }

    for (unsigned int c_index = 0; c_index < image->numcomps; c_index++)
    {
        int width = (int)image->comps[c_index].w;
        int height = (int)image->comps[c_index].h;
        int precision = (int)image->comps[c_index].prec;
        int bpp = (int)image->comps[c_index].bpp;
        int nr_bytes = ((precision + 7) & -8) / 8;

        printf(
            "Component %u characteristics: %d x %d, %s %d bit, %d bpp, %d byte\n",
            c_index, width, height,
            image->comps[c_index].sgnd == 1 ? "signed" : "unsigned",
            precision,
            bpp,
            nr_bytes
        );

        // Copy image data to the output uint8 numpy array
        // The decoded data type is OPJ_INT32, so we need to convert... ugh!
        int *ptr = image->comps[c_index].data;
        int mask = (1 << precision) - 1;

        union {
            unsigned short val;
            unsigned char vals[2];
        } u16;

        if (precision <= 8)
        {
            // 8-bit signed/unsigned
            for (int row = 0; row < height; row++)
            {
                for (int col = 0; col < width; col++)
                {
                    *out = (unsigned char)(*ptr & mask);
                    out++;
                    ptr++;
                }
            }
        }
        else if (precision <= 16)
        {
            // 16-bit
            for (int row = 0; row < height; row++)
            {
                for (int col = 0; col < width; col++)
                {
                    u16.val = (unsigned short)(*ptr & mask);
                    *out = u16.vals[0];
                    out++;
                    *out = u16.vals[1];
                    out++;
                    ptr++;
                }
            }
        }
        else
        {
            fprintf(stderr, "ERROR -> More than 16-bits per component not implemented\n");
            goto quick_exit;
        }


    }

    opj_destroy_codec(codec);
    opj_image_destroy(image);
    opj_stream_destroy(stream);

    return 1;

    quick_exit:
        //destroy_parameters(&parameters);
        if (codec)
            opj_destroy_codec(codec);
        if (image)
            opj_image_destroy(image);
        if (stream)
            opj_stream_destroy(stream);

        return -1;
}
