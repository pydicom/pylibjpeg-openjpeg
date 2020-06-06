
#include <stdlib.h>
//#include "Jpeg2K.h"
#include <../openjpeg/src/lib/openjp2/openjpeg.h>

#define BUFFER_SIZE OPJ_J2K_STREAM_CHUNK_SIZE


const char * OpenJpegVersion(void)
{
    // return the openjpeg version as char array
    return opj_version();
}


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


int decode() {
    // J2K stream
    opj_stream_t *stream = NULL;
    // struct defining image data and characteristics
    opj_image_t *image = NULL;
    // J2K codec
    opj_codec_t *codec = NULL;

    CodecState *state;

    // Creates an abstract input stream; allocates memory
    stream = opj_stream_create(BUFFER_SIZE, OPJ_TRUE);

    if (!stream) {
        state->error_code = IMAGING_CODEC_BROKEN;
        state->state = J2K_STATE_FAILED;
        goto quick_exit;
    }

    quick_exit:
       if (codec)
           opj_destroy_codec(codec);
       if (image)
           opj_image_destroy(image);
       if (stream)
           opj_stream_destroy(stream);

       return -1;

    return 1;
}
