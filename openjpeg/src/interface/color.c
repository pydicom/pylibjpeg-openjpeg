/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2002-2014, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2014, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux
 * Copyright (c) 2003-2014, Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "openjpeg.h"
#include "color.h"


 /*--------------------------------------------------------
 Matrix for sYCC, Amendment 1 to IEC 61966-2-1

 Y :   0.299   0.587    0.114   :R
 Cb:  -0.1687 -0.3312   0.5     :G
 Cr:   0.5    -0.4187  -0.0812  :B

 Inverse:

 R: 1        -3.68213e-05    1.40199      :Y
 G: 1.00003  -0.344125      -0.714128     :Cb - 2^(prec - 1)
 B: 0.999823  1.77204       -8.04142e-06  :Cr - 2^(prec - 1)

 -----------------------------------------------------------*/
static void sycc_to_rgb(
    int offset, int upb, int y, int cb, int cr,
    int *out_r, int *out_g, int *out_b
)
{
    int r, g, b;

    cb -= offset;
    cr -= offset;
    r = y + (int)(1.402 * (float)cr);
    if (r < 0)
    {
        r = 0;
    } else if (r > upb)
    {
        r = upb;
    }
    *out_r = r;

    g = y - (int)(0.344 * (float)cb + 0.714 * (float)cr);
    if (g < 0)
    {
        g = 0;
    } else if (g > upb)
    {
        g = upb;
    }
    *out_g = g;

    b = y + (int)(1.772 * (float)cb);
    if (b < 0)
    {
        b = 0;
    } else if (b > upb)
    {
        b = upb;
    }
    *out_b = b;
}

static void sycc444_to_rgb(opj_image_t *img)
{
    int *d0, *d1, *d2, *r, *g, *b;
    const int *y, *cb, *cr;
    size_t maxw, maxh, max, i;
    int offset, upb;

    upb = (int)img->comps[0].prec;
    offset = 1 << (upb - 1);
    upb = (1 << upb) - 1;

    maxw = (size_t)img->comps[0].w;
    maxh = (size_t)img->comps[0].h;
    max = maxw * maxh;

    y = img->comps[0].data;
    cb = img->comps[1].data;
    cr = img->comps[2].data;

    d0 = r = (int*)opj_image_data_alloc(sizeof(int) * max);
    d1 = g = (int*)opj_image_data_alloc(sizeof(int) * max);
    d2 = b = (int*)opj_image_data_alloc(sizeof(int) * max);

    if (r == NULL || g == NULL || b == NULL) {
        goto fails;
    }

    for (i = 0U; i < max; ++i) {
        sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);
        ++y;
        ++cb;
        ++cr;
        ++r;
        ++g;
        ++b;
    }
    opj_image_data_free(img->comps[0].data);
    img->comps[0].data = d0;
    opj_image_data_free(img->comps[1].data);
    img->comps[1].data = d1;
    opj_image_data_free(img->comps[2].data);
    img->comps[2].data = d2;
    img->color_space = OPJ_CLRSPC_SRGB;
    return;

fails:
    opj_image_data_free(r);
    opj_image_data_free(g);
    opj_image_data_free(b);
}


static void sycc422_to_rgb(opj_image_t *img)
{
    int *d0, *d1, *d2, *r, *g, *b;
    const int *y, *cb, *cr;
    size_t maxw, maxh, max, offx, loopmaxw;
    int offset, upb;
    size_t i;

    upb = (int)img->comps[0].prec;
    offset = 1 << (upb - 1);
    upb = (1 << upb) - 1;

    maxw = (size_t)img->comps[0].w;
    maxh = (size_t)img->comps[0].h;
    max = maxw * maxh;

    y = img->comps[0].data;
    cb = img->comps[1].data;
    cr = img->comps[2].data;

    d0 = r = (int*)opj_image_data_alloc(sizeof(int) * max);
    d1 = g = (int*)opj_image_data_alloc(sizeof(int) * max);
    d2 = b = (int*)opj_image_data_alloc(sizeof(int) * max);

    if (r == NULL || g == NULL || b == NULL) {
        goto fails;
    }

    /* if img->x0 is odd, then first column shall use Cb/Cr = 0 */
    offx = img->x0 & 1U;
    loopmaxw = maxw - offx;

    for (i = 0U; i < maxh; ++i) {
        size_t j;

        if (offx > 0U) {
            sycc_to_rgb(offset, upb, *y, 0, 0, r, g, b);
            ++y;
            ++r;
            ++g;
            ++b;
        }

        for (j = 0U; j < (loopmaxw & ~(size_t)1U); j += 2U) {
            sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);
            ++y;
            ++r;
            ++g;
            ++b;
            sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);
            ++y;
            ++r;
            ++g;
            ++b;
            ++cb;
            ++cr;
        }
        if (j < loopmaxw) {
            sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);
            ++y;
            ++r;
            ++g;
            ++b;
            ++cb;
            ++cr;
        }
    }

    opj_image_data_free(img->comps[0].data);
    img->comps[0].data = d0;
    opj_image_data_free(img->comps[1].data);
    img->comps[1].data = d1;
    opj_image_data_free(img->comps[2].data);
    img->comps[2].data = d2;

    img->comps[1].w = img->comps[2].w = img->comps[0].w;
    img->comps[1].h = img->comps[2].h = img->comps[0].h;
    img->comps[1].dx = img->comps[2].dx = img->comps[0].dx;
    img->comps[1].dy = img->comps[2].dy = img->comps[0].dy;
    img->color_space = OPJ_CLRSPC_SRGB;
    return;

fails:
    opj_image_data_free(r);
    opj_image_data_free(g);
    opj_image_data_free(b);
}


static void sycc420_to_rgb(opj_image_t *img)
{
    int *d0, *d1, *d2, *r, *g, *b, *nr, *ng, *nb;
    const int *y, *cb, *cr, *ny;
    size_t maxw, maxh, max, offx, loopmaxw, offy, loopmaxh;
    int offset, upb;
    size_t i;

    upb = (int)img->comps[0].prec;
    offset = 1 << (upb - 1);
    upb = (1 << upb) - 1;

    maxw = (size_t)img->comps[0].w;
    maxh = (size_t)img->comps[0].h;
    max = maxw * maxh;

    y = img->comps[0].data;
    cb = img->comps[1].data;
    cr = img->comps[2].data;

    d0 = r = (int*)opj_image_data_alloc(sizeof(int) * max);
    d1 = g = (int*)opj_image_data_alloc(sizeof(int) * max);
    d2 = b = (int*)opj_image_data_alloc(sizeof(int) * max);

    if (r == NULL || g == NULL || b == NULL) {
        goto fails;
    }

    /* if img->x0 is odd, then first column shall use Cb/Cr = 0 */
    offx = img->x0 & 1U;
    loopmaxw = maxw - offx;
    /* if img->y0 is odd, then first line shall use Cb/Cr = 0 */
    offy = img->y0 & 1U;
    loopmaxh = maxh - offy;

    if (offy > 0U) {
        size_t j;

        for (j = 0; j < maxw; ++j) {
            sycc_to_rgb(offset, upb, *y, 0, 0, r, g, b);
            ++y;
            ++r;
            ++g;
            ++b;
        }
    }

    for (i = 0U; i < (loopmaxh & ~(size_t)1U); i += 2U) {
        size_t j;

        ny = y + maxw;
        nr = r + maxw;
        ng = g + maxw;
        nb = b + maxw;

        if (offx > 0U) {
            sycc_to_rgb(offset, upb, *y, 0, 0, r, g, b);
            ++y;
            ++r;
            ++g;
            ++b;
            sycc_to_rgb(offset, upb, *ny, *cb, *cr, nr, ng, nb);
            ++ny;
            ++nr;
            ++ng;
            ++nb;
        }

        for (j = 0; j < (loopmaxw & ~(size_t)1U); j += 2U) {
            sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);
            ++y;
            ++r;
            ++g;
            ++b;
            sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);
            ++y;
            ++r;
            ++g;
            ++b;

            sycc_to_rgb(offset, upb, *ny, *cb, *cr, nr, ng, nb);
            ++ny;
            ++nr;
            ++ng;
            ++nb;
            sycc_to_rgb(offset, upb, *ny, *cb, *cr, nr, ng, nb);
            ++ny;
            ++nr;
            ++ng;
            ++nb;
            ++cb;
            ++cr;
        }
        if (j < loopmaxw) {
            sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);
            ++y;
            ++r;
            ++g;
            ++b;

            sycc_to_rgb(offset, upb, *ny, *cb, *cr, nr, ng, nb);
            ++ny;
            ++nr;
            ++ng;
            ++nb;
            ++cb;
            ++cr;
        }
        y += maxw;
        r += maxw;
        g += maxw;
        b += maxw;
    }
    if (i < loopmaxh) {
        size_t j;

        for (j = 0U; j < (maxw & ~(size_t)1U); j += 2U) {
            sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);

            ++y;
            ++r;
            ++g;
            ++b;

            sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);

            ++y;
            ++r;
            ++g;
            ++b;
            ++cb;
            ++cr;
        }
        if (j < maxw) {
            sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);
        }
    }

    opj_image_data_free(img->comps[0].data);
    img->comps[0].data = d0;
    opj_image_data_free(img->comps[1].data);
    img->comps[1].data = d1;
    opj_image_data_free(img->comps[2].data);
    img->comps[2].data = d2;

    img->comps[1].w = img->comps[2].w = img->comps[0].w;
    img->comps[1].h = img->comps[2].h = img->comps[0].h;
    img->comps[1].dx = img->comps[2].dx = img->comps[0].dx;
    img->comps[1].dy = img->comps[2].dy = img->comps[0].dy;
    img->color_space = OPJ_CLRSPC_SRGB;
    return;

fails:
    opj_image_data_free(r);
    opj_image_data_free(g);
    opj_image_data_free(b);
}


void color_sycc_to_rgb(opj_image_t *img)
{
    if (img->numcomps < 3) {
        img->color_space = OPJ_CLRSPC_GRAY;
        return;
    }

    if (
        (img->comps[0].dx == 1) && (img->comps[0].dy == 1)
        && (img->comps[1].dx == 2) && (img->comps[1].dy == 2)
        && (img->comps[2].dx == 2) && (img->comps[2].dy == 2)
    )
    {
        /* horizontal and vertical sub-sample */
        sycc420_to_rgb(img);
    } else if (
        (img->comps[0].dx == 1) && (img->comps[0].dy == 1)
        && (img->comps[1].dx == 2) && (img->comps[1].dy == 1)
        && (img->comps[2].dx == 2) && (img->comps[2].dy == 1)
    )
    {
        /* horizontal sub-sample only */
        sycc422_to_rgb(img);
    } else if (
        (img->comps[0].dx == 1) && (img->comps[0].dy == 1)
        && (img->comps[1].dx == 1) && (img->comps[1].dy == 1)
        && (img->comps[2].dx == 1) && (img->comps[2].dy == 1)
    )
    {
        /* no sub-sample */
        sycc444_to_rgb(img);
    } else {
        return;
    }
}
