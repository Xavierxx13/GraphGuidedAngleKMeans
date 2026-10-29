#ifndef AKM_MYKM_KERNELS_H
#define AKM_MYKM_KERNELS_H

#include <immintrin.h>
#include <cstdint>

namespace mykm {

inline void micro_4x1_fused(const float* x0, const float* x1, const float* x2, const float* x3,
                            const float* cen, int dim, float cn, int cj,
                            __m512 twos, __mmask16 row_mask,
                            __m512& best_vec, __m512i& bestj_vec) {
    __m512 acc0=_mm512_setzero_ps(), acc1=_mm512_setzero_ps(),
           acc2=_mm512_setzero_ps(), acc3=_mm512_setzero_ps();
    int dd = 0;
    for (; dd + 15 < dim; dd += 16) {
        __m512 cv = _mm512_loadu_ps(cen + dd);
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x0 + dd), cv, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x1 + dd), cv, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(x2 + dd), cv, acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(x3 + dd), cv, acc3);
    }
    float s0 = _mm512_reduce_add_ps(acc0), s1 = _mm512_reduce_add_ps(acc1),
          s2 = _mm512_reduce_add_ps(acc2), s3 = _mm512_reduce_add_ps(acc3);
    for (int kk = dd; kk < dim; kk++) {
        float cv = cen[kk];
        s0 += x0[kk]*cv; s1 += x1[kk]*cv; s2 += x2[kk]*cv; s3 += x3[kk]*cv;
    }

    __m512  dots_vec  = _mm512_zextps128_ps512(_mm_set_ps(s3, s2, s1, s0));
    __m512  dist_vec  = _mm512_fnmadd_ps(dots_vec, twos, _mm512_set1_ps(cn));   
    __mmask16 improve  = _mm512_cmp_ps_mask(dist_vec, best_vec, _CMP_LT_OS) & row_mask;
    best_vec  = _mm512_mask_blend_ps(improve, best_vec, dist_vec);
    bestj_vec = _mm512_mask_blend_epi32(improve, bestj_vec, _mm512_set1_epi32(cj));
}

inline void micro_8x1_fused(const float* x8, const float* cen, int dim,
                            float cn, int cj, __m512 twos, __mmask16 row_mask,
                            __m512& best_vec, __m512i& bestj_vec) {
    const float *x0=x8, *x1=x8+(size_t)1*dim, *x2=x8+(size_t)2*dim, *x3=x8+(size_t)3*dim,
                *x4=x8+(size_t)4*dim, *x5=x8+(size_t)5*dim, *x6=x8+(size_t)6*dim, *x7=x8+(size_t)7*dim;
    __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps(),
           a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps(),
           a4=_mm512_setzero_ps(), a5=_mm512_setzero_ps(),
           a6=_mm512_setzero_ps(), a7=_mm512_setzero_ps();
    int dd = 0;
    for (; dd + 15 < dim; dd += 16) {
        __m512 cv = _mm512_loadu_ps(cen + dd);                   
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(x0 + dd), cv, a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(x1 + dd), cv, a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(x2 + dd), cv, a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(x3 + dd), cv, a3);
        a4 = _mm512_fmadd_ps(_mm512_loadu_ps(x4 + dd), cv, a4);
        a5 = _mm512_fmadd_ps(_mm512_loadu_ps(x5 + dd), cv, a5);
        a6 = _mm512_fmadd_ps(_mm512_loadu_ps(x6 + dd), cv, a6);
        a7 = _mm512_fmadd_ps(_mm512_loadu_ps(x7 + dd), cv, a7);
    }
    float s0=_mm512_reduce_add_ps(a0), s1=_mm512_reduce_add_ps(a1),
          s2=_mm512_reduce_add_ps(a2), s3=_mm512_reduce_add_ps(a3),
          s4=_mm512_reduce_add_ps(a4), s5=_mm512_reduce_add_ps(a5),
          s6=_mm512_reduce_add_ps(a6), s7=_mm512_reduce_add_ps(a7);
    for (int kk = dd; kk < dim; kk++) {                          
        float cv = cen[kk];
        s0 += x0[kk]*cv; s1 += x1[kk]*cv; s2 += x2[kk]*cv; s3 += x3[kk]*cv;
        s4 += x4[kk]*cv; s5 += x5[kk]*cv; s6 += x6[kk]*cv; s7 += x7[kk]*cv;
    }
    __m512  dots = _mm512_zextps256_ps512(_mm256_set_ps(s7, s6, s5, s4, s3, s2, s1, s0));  
    __m512  dist = _mm512_fnmadd_ps(dots, twos, _mm512_set1_ps(cn));   
    __mmask16 imp = _mm512_cmp_ps_mask(dist, best_vec, _CMP_LT_OS) & row_mask;
    best_vec  = _mm512_mask_blend_ps(imp, best_vec, dist);
    bestj_vec = _mm512_mask_blend_epi32(imp, bestj_vec, _mm512_set1_epi32(cj));
}

static inline __attribute__((always_inline)) void micro_1x1_dots(const float* x0, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(); int dd=0;
    for (; dd+15<dim; dd+=16) { __m512 cv=_mm512_loadu_ps(cen+dd); a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0); }
    out[0]=_mm512_reduce_add_ps(a0); for(int kk=dd;kk<dim;kk++) out[0]+=x0[kk]*cen[kk];
}
static inline __attribute__((always_inline)) void micro_2x1_dots(const float* x0,const float* x1, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(); int dd=0;
    for (; dd+15<dim; dd+=16) { __m512 cv=_mm512_loadu_ps(cen+dd); a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0); a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1); }
    out[0]=_mm512_reduce_add_ps(a0); out[1]=_mm512_reduce_add_ps(a1);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk]; out[0]+=x0[kk]*cv; out[1]+=x1[kk]*cv;}
}
static inline __attribute__((always_inline)) void micro_3x1_dots(const float* x0,const float* x1,const float* x2, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(); int dd=0;
    for (; dd+15<dim; dd+=16) { __m512 cv=_mm512_loadu_ps(cen+dd); a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0); a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1); a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2); }
    out[0]=_mm512_reduce_add_ps(a0); out[1]=_mm512_reduce_add_ps(a1); out[2]=_mm512_reduce_add_ps(a2);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk]; out[0]+=x0[kk]*cv; out[1]+=x1[kk]*cv; out[2]+=x2[kk]*cv;}
}
static inline __attribute__((always_inline)) void micro_4x1_dots(const float* x0,const float* x1,const float* x2,const float* x3, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(); int dd=0;
    for (; dd+15<dim; dd+=16) { __m512 cv=_mm512_loadu_ps(cen+dd); a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0); a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1); a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2); a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3); }
    out[0]=_mm512_reduce_add_ps(a0); out[1]=_mm512_reduce_add_ps(a1); out[2]=_mm512_reduce_add_ps(a2); out[3]=_mm512_reduce_add_ps(a3);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk]; out[0]+=x0[kk]*cv; out[1]+=x1[kk]*cv; out[2]+=x2[kk]*cv; out[3]+=x3[kk]*cv;}
}
static inline __attribute__((always_inline)) void micro_5x1_dots(const float* x0,const float* x1,const float* x2,const float* x3,const float* x4, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),a4=_mm512_setzero_ps(); int dd=0;
    for (; dd+15<dim; dd+=16) { __m512 cv=_mm512_loadu_ps(cen+dd); a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0); a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1); a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2); a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3); a4=_mm512_fmadd_ps(_mm512_loadu_ps(x4+dd),cv,a4); }
    out[0]=_mm512_reduce_add_ps(a0); out[1]=_mm512_reduce_add_ps(a1); out[2]=_mm512_reduce_add_ps(a2); out[3]=_mm512_reduce_add_ps(a3); out[4]=_mm512_reduce_add_ps(a4);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk]; out[0]+=x0[kk]*cv; out[1]+=x1[kk]*cv; out[2]+=x2[kk]*cv; out[3]+=x3[kk]*cv; out[4]+=x4[kk]*cv;}
}
static inline __attribute__((always_inline)) void micro_6x1_dots(const float* x0,const float* x1,const float* x2,const float* x3,const float* x4,const float* x5, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),a4=_mm512_setzero_ps(),a5=_mm512_setzero_ps(); int dd=0;
    for (; dd+15<dim; dd+=16) { __m512 cv=_mm512_loadu_ps(cen+dd); a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0); a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1); a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2); a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3); a4=_mm512_fmadd_ps(_mm512_loadu_ps(x4+dd),cv,a4); a5=_mm512_fmadd_ps(_mm512_loadu_ps(x5+dd),cv,a5); }
    out[0]=_mm512_reduce_add_ps(a0); out[1]=_mm512_reduce_add_ps(a1); out[2]=_mm512_reduce_add_ps(a2); out[3]=_mm512_reduce_add_ps(a3); out[4]=_mm512_reduce_add_ps(a4); out[5]=_mm512_reduce_add_ps(a5);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk]; out[0]+=x0[kk]*cv; out[1]+=x1[kk]*cv; out[2]+=x2[kk]*cv; out[3]+=x3[kk]*cv; out[4]+=x4[kk]*cv; out[5]+=x5[kk]*cv;}
}
static inline __attribute__((always_inline)) void micro_7x1_dots(const float* x0,const float* x1,const float* x2,const float* x3,const float* x4,const float* x5,const float* x6, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),a4=_mm512_setzero_ps(),a5=_mm512_setzero_ps(),a6=_mm512_setzero_ps(); int dd=0;
    for (; dd+15<dim; dd+=16) { __m512 cv=_mm512_loadu_ps(cen+dd); a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0); a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1); a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2); a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3); a4=_mm512_fmadd_ps(_mm512_loadu_ps(x4+dd),cv,a4); a5=_mm512_fmadd_ps(_mm512_loadu_ps(x5+dd),cv,a5); a6=_mm512_fmadd_ps(_mm512_loadu_ps(x6+dd),cv,a6); }
    out[0]=_mm512_reduce_add_ps(a0); out[1]=_mm512_reduce_add_ps(a1); out[2]=_mm512_reduce_add_ps(a2); out[3]=_mm512_reduce_add_ps(a3); out[4]=_mm512_reduce_add_ps(a4); out[5]=_mm512_reduce_add_ps(a5); out[6]=_mm512_reduce_add_ps(a6);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk]; out[0]+=x0[kk]*cv; out[1]+=x1[kk]*cv; out[2]+=x2[kk]*cv; out[3]+=x3[kk]*cv; out[4]+=x4[kk]*cv; out[5]+=x5[kk]*cv; out[6]+=x6[kk]*cv;}
}
static inline __attribute__((always_inline)) void micro_8x1_dots(const float* x0,const float* x1,const float* x2,const float* x3,const float* x4,const float* x5,const float* x6,const float* x7, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),a4=_mm512_setzero_ps(),a5=_mm512_setzero_ps(),a6=_mm512_setzero_ps(),a7=_mm512_setzero_ps(); int dd=0;
    for (; dd+15<dim; dd+=16) { __m512 cv=_mm512_loadu_ps(cen+dd); a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0); a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1); a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2); a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3); a4=_mm512_fmadd_ps(_mm512_loadu_ps(x4+dd),cv,a4); a5=_mm512_fmadd_ps(_mm512_loadu_ps(x5+dd),cv,a5); a6=_mm512_fmadd_ps(_mm512_loadu_ps(x6+dd),cv,a6); a7=_mm512_fmadd_ps(_mm512_loadu_ps(x7+dd),cv,a7); }
    out[0]=_mm512_reduce_add_ps(a0); out[1]=_mm512_reduce_add_ps(a1); out[2]=_mm512_reduce_add_ps(a2); out[3]=_mm512_reduce_add_ps(a3); out[4]=_mm512_reduce_add_ps(a4); out[5]=_mm512_reduce_add_ps(a5); out[6]=_mm512_reduce_add_ps(a6); out[7]=_mm512_reduce_add_ps(a7);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk]; out[0]+=x0[kk]*cv; out[1]+=x1[kk]*cv; out[2]+=x2[kk]*cv; out[3]+=x3[kk]*cv; out[4]+=x4[kk]*cv; out[5]+=x5[kk]*cv; out[6]+=x6[kk]*cv; out[7]+=x7[kk]*cv;}
}

static inline __attribute__((always_inline)) void micro_9x1_dots(const float* x0,const float* x1,const float* x2,const float* x3,const float* x4,const float* x5,const float* x6,const float* x7,const float* x8, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),a4=_mm512_setzero_ps(),a5=_mm512_setzero_ps(),a6=_mm512_setzero_ps(),a7=_mm512_setzero_ps(),a8=_mm512_setzero_ps(); int dd=0;
    for(;dd+15<dim;dd+=16){__m512 cv=_mm512_loadu_ps(cen+dd);a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0);a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1);a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2);a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3);a4=_mm512_fmadd_ps(_mm512_loadu_ps(x4+dd),cv,a4);a5=_mm512_fmadd_ps(_mm512_loadu_ps(x5+dd),cv,a5);a6=_mm512_fmadd_ps(_mm512_loadu_ps(x6+dd),cv,a6);a7=_mm512_fmadd_ps(_mm512_loadu_ps(x7+dd),cv,a7);a8=_mm512_fmadd_ps(_mm512_loadu_ps(x8+dd),cv,a8);}
    out[0]=_mm512_reduce_add_ps(a0);out[1]=_mm512_reduce_add_ps(a1);out[2]=_mm512_reduce_add_ps(a2);out[3]=_mm512_reduce_add_ps(a3);out[4]=_mm512_reduce_add_ps(a4);out[5]=_mm512_reduce_add_ps(a5);out[6]=_mm512_reduce_add_ps(a6);out[7]=_mm512_reduce_add_ps(a7);out[8]=_mm512_reduce_add_ps(a8);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk];out[0]+=x0[kk]*cv;out[1]+=x1[kk]*cv;out[2]+=x2[kk]*cv;out[3]+=x3[kk]*cv;out[4]+=x4[kk]*cv;out[5]+=x5[kk]*cv;out[6]+=x6[kk]*cv;out[7]+=x7[kk]*cv;out[8]+=x8[kk]*cv;}}

static inline __attribute__((always_inline)) void micro_10x1_dots(const float* x0,const float* x1,const float* x2,const float* x3,const float* x4,const float* x5,const float* x6,const float* x7,const float* x8,const float* x9, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),a4=_mm512_setzero_ps(),a5=_mm512_setzero_ps(),a6=_mm512_setzero_ps(),a7=_mm512_setzero_ps(),a8=_mm512_setzero_ps(),a9=_mm512_setzero_ps(); int dd=0;
    for(;dd+15<dim;dd+=16){__m512 cv=_mm512_loadu_ps(cen+dd);a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0);a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1);a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2);a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3);a4=_mm512_fmadd_ps(_mm512_loadu_ps(x4+dd),cv,a4);a5=_mm512_fmadd_ps(_mm512_loadu_ps(x5+dd),cv,a5);a6=_mm512_fmadd_ps(_mm512_loadu_ps(x6+dd),cv,a6);a7=_mm512_fmadd_ps(_mm512_loadu_ps(x7+dd),cv,a7);a8=_mm512_fmadd_ps(_mm512_loadu_ps(x8+dd),cv,a8);a9=_mm512_fmadd_ps(_mm512_loadu_ps(x9+dd),cv,a9);}
    out[0]=_mm512_reduce_add_ps(a0);out[1]=_mm512_reduce_add_ps(a1);out[2]=_mm512_reduce_add_ps(a2);out[3]=_mm512_reduce_add_ps(a3);out[4]=_mm512_reduce_add_ps(a4);out[5]=_mm512_reduce_add_ps(a5);out[6]=_mm512_reduce_add_ps(a6);out[7]=_mm512_reduce_add_ps(a7);out[8]=_mm512_reduce_add_ps(a8);out[9]=_mm512_reduce_add_ps(a9);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk];out[0]+=x0[kk]*cv;out[1]+=x1[kk]*cv;out[2]+=x2[kk]*cv;out[3]+=x3[kk]*cv;out[4]+=x4[kk]*cv;out[5]+=x5[kk]*cv;out[6]+=x6[kk]*cv;out[7]+=x7[kk]*cv;out[8]+=x8[kk]*cv;out[9]+=x9[kk]*cv;}}

static inline __attribute__((always_inline)) void micro_11x1_dots(const float* x0,const float* x1,const float* x2,const float* x3,const float* x4,const float* x5,const float* x6,const float* x7,const float* x8,const float* x9,const float* x10, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),a4=_mm512_setzero_ps(),a5=_mm512_setzero_ps(),a6=_mm512_setzero_ps(),a7=_mm512_setzero_ps(),a8=_mm512_setzero_ps(),a9=_mm512_setzero_ps(),a10=_mm512_setzero_ps(); int dd=0;
    for(;dd+15<dim;dd+=16){__m512 cv=_mm512_loadu_ps(cen+dd);a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0);a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1);a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2);a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3);a4=_mm512_fmadd_ps(_mm512_loadu_ps(x4+dd),cv,a4);a5=_mm512_fmadd_ps(_mm512_loadu_ps(x5+dd),cv,a5);a6=_mm512_fmadd_ps(_mm512_loadu_ps(x6+dd),cv,a6);a7=_mm512_fmadd_ps(_mm512_loadu_ps(x7+dd),cv,a7);a8=_mm512_fmadd_ps(_mm512_loadu_ps(x8+dd),cv,a8);a9=_mm512_fmadd_ps(_mm512_loadu_ps(x9+dd),cv,a9);a10=_mm512_fmadd_ps(_mm512_loadu_ps(x10+dd),cv,a10);}
    out[0]=_mm512_reduce_add_ps(a0);out[1]=_mm512_reduce_add_ps(a1);out[2]=_mm512_reduce_add_ps(a2);out[3]=_mm512_reduce_add_ps(a3);out[4]=_mm512_reduce_add_ps(a4);out[5]=_mm512_reduce_add_ps(a5);out[6]=_mm512_reduce_add_ps(a6);out[7]=_mm512_reduce_add_ps(a7);out[8]=_mm512_reduce_add_ps(a8);out[9]=_mm512_reduce_add_ps(a9);out[10]=_mm512_reduce_add_ps(a10);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk];out[0]+=x0[kk]*cv;out[1]+=x1[kk]*cv;out[2]+=x2[kk]*cv;out[3]+=x3[kk]*cv;out[4]+=x4[kk]*cv;out[5]+=x5[kk]*cv;out[6]+=x6[kk]*cv;out[7]+=x7[kk]*cv;out[8]+=x8[kk]*cv;out[9]+=x9[kk]*cv;out[10]+=x10[kk]*cv;}}

static inline __attribute__((always_inline)) void micro_12x1_dots(const float* x0,const float* x1,const float* x2,const float* x3,const float* x4,const float* x5,const float* x6,const float* x7,const float* x8,const float* x9,const float* x10,const float* x11, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),a4=_mm512_setzero_ps(),a5=_mm512_setzero_ps(),a6=_mm512_setzero_ps(),a7=_mm512_setzero_ps(),a8=_mm512_setzero_ps(),a9=_mm512_setzero_ps(),a10=_mm512_setzero_ps(),a11=_mm512_setzero_ps(); int dd=0;
    for(;dd+15<dim;dd+=16){__m512 cv=_mm512_loadu_ps(cen+dd);a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0);a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1);a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2);a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3);a4=_mm512_fmadd_ps(_mm512_loadu_ps(x4+dd),cv,a4);a5=_mm512_fmadd_ps(_mm512_loadu_ps(x5+dd),cv,a5);a6=_mm512_fmadd_ps(_mm512_loadu_ps(x6+dd),cv,a6);a7=_mm512_fmadd_ps(_mm512_loadu_ps(x7+dd),cv,a7);a8=_mm512_fmadd_ps(_mm512_loadu_ps(x8+dd),cv,a8);a9=_mm512_fmadd_ps(_mm512_loadu_ps(x9+dd),cv,a9);a10=_mm512_fmadd_ps(_mm512_loadu_ps(x10+dd),cv,a10);a11=_mm512_fmadd_ps(_mm512_loadu_ps(x11+dd),cv,a11);}
    out[0]=_mm512_reduce_add_ps(a0);out[1]=_mm512_reduce_add_ps(a1);out[2]=_mm512_reduce_add_ps(a2);out[3]=_mm512_reduce_add_ps(a3);out[4]=_mm512_reduce_add_ps(a4);out[5]=_mm512_reduce_add_ps(a5);out[6]=_mm512_reduce_add_ps(a6);out[7]=_mm512_reduce_add_ps(a7);out[8]=_mm512_reduce_add_ps(a8);out[9]=_mm512_reduce_add_ps(a9);out[10]=_mm512_reduce_add_ps(a10);out[11]=_mm512_reduce_add_ps(a11);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk];out[0]+=x0[kk]*cv;out[1]+=x1[kk]*cv;out[2]+=x2[kk]*cv;out[3]+=x3[kk]*cv;out[4]+=x4[kk]*cv;out[5]+=x5[kk]*cv;out[6]+=x6[kk]*cv;out[7]+=x7[kk]*cv;out[8]+=x8[kk]*cv;out[9]+=x9[kk]*cv;out[10]+=x10[kk]*cv;out[11]+=x11[kk]*cv;}}

static inline __attribute__((always_inline)) void micro_13x1_dots(const float* x0,const float* x1,const float* x2,const float* x3,const float* x4,const float* x5,const float* x6,const float* x7,const float* x8,const float* x9,const float* x10,const float* x11,const float* x12, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),a4=_mm512_setzero_ps(),a5=_mm512_setzero_ps(),a6=_mm512_setzero_ps(),a7=_mm512_setzero_ps(),a8=_mm512_setzero_ps(),a9=_mm512_setzero_ps(),a10=_mm512_setzero_ps(),a11=_mm512_setzero_ps(),a12=_mm512_setzero_ps(); int dd=0;
    for(;dd+15<dim;dd+=16){__m512 cv=_mm512_loadu_ps(cen+dd);a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0);a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1);a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2);a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3);a4=_mm512_fmadd_ps(_mm512_loadu_ps(x4+dd),cv,a4);a5=_mm512_fmadd_ps(_mm512_loadu_ps(x5+dd),cv,a5);a6=_mm512_fmadd_ps(_mm512_loadu_ps(x6+dd),cv,a6);a7=_mm512_fmadd_ps(_mm512_loadu_ps(x7+dd),cv,a7);a8=_mm512_fmadd_ps(_mm512_loadu_ps(x8+dd),cv,a8);a9=_mm512_fmadd_ps(_mm512_loadu_ps(x9+dd),cv,a9);a10=_mm512_fmadd_ps(_mm512_loadu_ps(x10+dd),cv,a10);a11=_mm512_fmadd_ps(_mm512_loadu_ps(x11+dd),cv,a11);a12=_mm512_fmadd_ps(_mm512_loadu_ps(x12+dd),cv,a12);}
    out[0]=_mm512_reduce_add_ps(a0);out[1]=_mm512_reduce_add_ps(a1);out[2]=_mm512_reduce_add_ps(a2);out[3]=_mm512_reduce_add_ps(a3);out[4]=_mm512_reduce_add_ps(a4);out[5]=_mm512_reduce_add_ps(a5);out[6]=_mm512_reduce_add_ps(a6);out[7]=_mm512_reduce_add_ps(a7);out[8]=_mm512_reduce_add_ps(a8);out[9]=_mm512_reduce_add_ps(a9);out[10]=_mm512_reduce_add_ps(a10);out[11]=_mm512_reduce_add_ps(a11);out[12]=_mm512_reduce_add_ps(a12);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk];out[0]+=x0[kk]*cv;out[1]+=x1[kk]*cv;out[2]+=x2[kk]*cv;out[3]+=x3[kk]*cv;out[4]+=x4[kk]*cv;out[5]+=x5[kk]*cv;out[6]+=x6[kk]*cv;out[7]+=x7[kk]*cv;out[8]+=x8[kk]*cv;out[9]+=x9[kk]*cv;out[10]+=x10[kk]*cv;out[11]+=x11[kk]*cv;out[12]+=x12[kk]*cv;}}

static inline __attribute__((always_inline)) void micro_14x1_dots(const float* x0,const float* x1,const float* x2,const float* x3,const float* x4,const float* x5,const float* x6,const float* x7,const float* x8,const float* x9,const float* x10,const float* x11,const float* x12,const float* x13, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),a4=_mm512_setzero_ps(),a5=_mm512_setzero_ps(),a6=_mm512_setzero_ps(),a7=_mm512_setzero_ps(),a8=_mm512_setzero_ps(),a9=_mm512_setzero_ps(),a10=_mm512_setzero_ps(),a11=_mm512_setzero_ps(),a12=_mm512_setzero_ps(),a13=_mm512_setzero_ps(); int dd=0;
    for(;dd+15<dim;dd+=16){__m512 cv=_mm512_loadu_ps(cen+dd);a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0);a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1);a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2);a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3);a4=_mm512_fmadd_ps(_mm512_loadu_ps(x4+dd),cv,a4);a5=_mm512_fmadd_ps(_mm512_loadu_ps(x5+dd),cv,a5);a6=_mm512_fmadd_ps(_mm512_loadu_ps(x6+dd),cv,a6);a7=_mm512_fmadd_ps(_mm512_loadu_ps(x7+dd),cv,a7);a8=_mm512_fmadd_ps(_mm512_loadu_ps(x8+dd),cv,a8);a9=_mm512_fmadd_ps(_mm512_loadu_ps(x9+dd),cv,a9);a10=_mm512_fmadd_ps(_mm512_loadu_ps(x10+dd),cv,a10);a11=_mm512_fmadd_ps(_mm512_loadu_ps(x11+dd),cv,a11);a12=_mm512_fmadd_ps(_mm512_loadu_ps(x12+dd),cv,a12);a13=_mm512_fmadd_ps(_mm512_loadu_ps(x13+dd),cv,a13);}
    out[0]=_mm512_reduce_add_ps(a0);out[1]=_mm512_reduce_add_ps(a1);out[2]=_mm512_reduce_add_ps(a2);out[3]=_mm512_reduce_add_ps(a3);out[4]=_mm512_reduce_add_ps(a4);out[5]=_mm512_reduce_add_ps(a5);out[6]=_mm512_reduce_add_ps(a6);out[7]=_mm512_reduce_add_ps(a7);out[8]=_mm512_reduce_add_ps(a8);out[9]=_mm512_reduce_add_ps(a9);out[10]=_mm512_reduce_add_ps(a10);out[11]=_mm512_reduce_add_ps(a11);out[12]=_mm512_reduce_add_ps(a12);out[13]=_mm512_reduce_add_ps(a13);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk];out[0]+=x0[kk]*cv;out[1]+=x1[kk]*cv;out[2]+=x2[kk]*cv;out[3]+=x3[kk]*cv;out[4]+=x4[kk]*cv;out[5]+=x5[kk]*cv;out[6]+=x6[kk]*cv;out[7]+=x7[kk]*cv;out[8]+=x8[kk]*cv;out[9]+=x9[kk]*cv;out[10]+=x10[kk]*cv;out[11]+=x11[kk]*cv;out[12]+=x12[kk]*cv;out[13]+=x13[kk]*cv;}}

static inline __attribute__((always_inline)) void micro_15x1_dots(const float* x0,const float* x1,const float* x2,const float* x3,const float* x4,const float* x5,const float* x6,const float* x7,const float* x8,const float* x9,const float* x10,const float* x11,const float* x12,const float* x13,const float* x14, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),a4=_mm512_setzero_ps(),a5=_mm512_setzero_ps(),a6=_mm512_setzero_ps(),a7=_mm512_setzero_ps(),a8=_mm512_setzero_ps(),a9=_mm512_setzero_ps(),a10=_mm512_setzero_ps(),a11=_mm512_setzero_ps(),a12=_mm512_setzero_ps(),a13=_mm512_setzero_ps(),a14=_mm512_setzero_ps(); int dd=0;
    for(;dd+15<dim;dd+=16){__m512 cv=_mm512_loadu_ps(cen+dd);a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0);a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1);a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2);a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3);a4=_mm512_fmadd_ps(_mm512_loadu_ps(x4+dd),cv,a4);a5=_mm512_fmadd_ps(_mm512_loadu_ps(x5+dd),cv,a5);a6=_mm512_fmadd_ps(_mm512_loadu_ps(x6+dd),cv,a6);a7=_mm512_fmadd_ps(_mm512_loadu_ps(x7+dd),cv,a7);a8=_mm512_fmadd_ps(_mm512_loadu_ps(x8+dd),cv,a8);a9=_mm512_fmadd_ps(_mm512_loadu_ps(x9+dd),cv,a9);a10=_mm512_fmadd_ps(_mm512_loadu_ps(x10+dd),cv,a10);a11=_mm512_fmadd_ps(_mm512_loadu_ps(x11+dd),cv,a11);a12=_mm512_fmadd_ps(_mm512_loadu_ps(x12+dd),cv,a12);a13=_mm512_fmadd_ps(_mm512_loadu_ps(x13+dd),cv,a13);a14=_mm512_fmadd_ps(_mm512_loadu_ps(x14+dd),cv,a14);}
    out[0]=_mm512_reduce_add_ps(a0);out[1]=_mm512_reduce_add_ps(a1);out[2]=_mm512_reduce_add_ps(a2);out[3]=_mm512_reduce_add_ps(a3);out[4]=_mm512_reduce_add_ps(a4);out[5]=_mm512_reduce_add_ps(a5);out[6]=_mm512_reduce_add_ps(a6);out[7]=_mm512_reduce_add_ps(a7);out[8]=_mm512_reduce_add_ps(a8);out[9]=_mm512_reduce_add_ps(a9);out[10]=_mm512_reduce_add_ps(a10);out[11]=_mm512_reduce_add_ps(a11);out[12]=_mm512_reduce_add_ps(a12);out[13]=_mm512_reduce_add_ps(a13);out[14]=_mm512_reduce_add_ps(a14);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk];out[0]+=x0[kk]*cv;out[1]+=x1[kk]*cv;out[2]+=x2[kk]*cv;out[3]+=x3[kk]*cv;out[4]+=x4[kk]*cv;out[5]+=x5[kk]*cv;out[6]+=x6[kk]*cv;out[7]+=x7[kk]*cv;out[8]+=x8[kk]*cv;out[9]+=x9[kk]*cv;out[10]+=x10[kk]*cv;out[11]+=x11[kk]*cv;out[12]+=x12[kk]*cv;out[13]+=x13[kk]*cv;out[14]+=x14[kk]*cv;}}

static inline __attribute__((always_inline)) void micro_16x1_dots(const float* x0,const float* x1,const float* x2,const float* x3,const float* x4,const float* x5,const float* x6,const float* x7,const float* x8,const float* x9,const float* x10,const float* x11,const float* x12,const float* x13,const float* x14,const float* x15, const float* cen, int dim, float* out) {
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps(),a4=_mm512_setzero_ps(),a5=_mm512_setzero_ps(),a6=_mm512_setzero_ps(),a7=_mm512_setzero_ps(),a8=_mm512_setzero_ps(),a9=_mm512_setzero_ps(),a10=_mm512_setzero_ps(),a11=_mm512_setzero_ps(),a12=_mm512_setzero_ps(),a13=_mm512_setzero_ps(),a14=_mm512_setzero_ps(),a15=_mm512_setzero_ps(); int dd=0;
    for(;dd+15<dim;dd+=16){__m512 cv=_mm512_loadu_ps(cen+dd);a0=_mm512_fmadd_ps(_mm512_loadu_ps(x0+dd),cv,a0);a1=_mm512_fmadd_ps(_mm512_loadu_ps(x1+dd),cv,a1);a2=_mm512_fmadd_ps(_mm512_loadu_ps(x2+dd),cv,a2);a3=_mm512_fmadd_ps(_mm512_loadu_ps(x3+dd),cv,a3);a4=_mm512_fmadd_ps(_mm512_loadu_ps(x4+dd),cv,a4);a5=_mm512_fmadd_ps(_mm512_loadu_ps(x5+dd),cv,a5);a6=_mm512_fmadd_ps(_mm512_loadu_ps(x6+dd),cv,a6);a7=_mm512_fmadd_ps(_mm512_loadu_ps(x7+dd),cv,a7);a8=_mm512_fmadd_ps(_mm512_loadu_ps(x8+dd),cv,a8);a9=_mm512_fmadd_ps(_mm512_loadu_ps(x9+dd),cv,a9);a10=_mm512_fmadd_ps(_mm512_loadu_ps(x10+dd),cv,a10);a11=_mm512_fmadd_ps(_mm512_loadu_ps(x11+dd),cv,a11);a12=_mm512_fmadd_ps(_mm512_loadu_ps(x12+dd),cv,a12);a13=_mm512_fmadd_ps(_mm512_loadu_ps(x13+dd),cv,a13);a14=_mm512_fmadd_ps(_mm512_loadu_ps(x14+dd),cv,a14);a15=_mm512_fmadd_ps(_mm512_loadu_ps(x15+dd),cv,a15);}
    out[0]=_mm512_reduce_add_ps(a0);out[1]=_mm512_reduce_add_ps(a1);out[2]=_mm512_reduce_add_ps(a2);out[3]=_mm512_reduce_add_ps(a3);out[4]=_mm512_reduce_add_ps(a4);out[5]=_mm512_reduce_add_ps(a5);out[6]=_mm512_reduce_add_ps(a6);out[7]=_mm512_reduce_add_ps(a7);out[8]=_mm512_reduce_add_ps(a8);out[9]=_mm512_reduce_add_ps(a9);out[10]=_mm512_reduce_add_ps(a10);out[11]=_mm512_reduce_add_ps(a11);out[12]=_mm512_reduce_add_ps(a12);out[13]=_mm512_reduce_add_ps(a13);out[14]=_mm512_reduce_add_ps(a14);out[15]=_mm512_reduce_add_ps(a15);
    for(int kk=dd;kk<dim;kk++){float cv=cen[kk];out[0]+=x0[kk]*cv;out[1]+=x1[kk]*cv;out[2]+=x2[kk]*cv;out[3]+=x3[kk]*cv;out[4]+=x4[kk]*cv;out[5]+=x5[kk]*cv;out[6]+=x6[kk]*cv;out[7]+=x7[kk]*cv;out[8]+=x8[kk]*cv;out[9]+=x9[kk]*cv;out[10]+=x10[kk]*cv;out[11]+=x11[kk]*cv;out[12]+=x12[kk]*cv;out[13]+=x13[kk]*cv;out[14]+=x14[kk]*cv;out[15]+=x15[kk]*cv;}}

inline float sum_sq_512(const float* a, int dim) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps(),
           acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    int d = 0;
    for (; d + 63 < dim; d += 64) {                 
        __m512 v0 = _mm512_loadu_ps(a + d),    v1 = _mm512_loadu_ps(a + d + 16),
               v2 = _mm512_loadu_ps(a + d + 32), v3 = _mm512_loadu_ps(a + d + 48);
        acc0 = _mm512_fmadd_ps(v0, v0, acc0); acc1 = _mm512_fmadd_ps(v1, v1, acc1);
        acc2 = _mm512_fmadd_ps(v2, v2, acc2); acc3 = _mm512_fmadd_ps(v3, v3, acc3);
    }
    for (; d + 15 < dim; d += 16) { __m512 v = _mm512_loadu_ps(a + d); acc0 = _mm512_fmadd_ps(v, v, acc0); }
    __m512 s = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    float res = _mm512_reduce_add_ps(s);
    for (; d < dim; d++) res += a[d] * a[d];
    return res;
}

inline void row_norm_sq(const float* M, float* out, int rows, int dim, int n_threads) {
    if (omp_in_parallel()) {
        for (int i = 0; i < rows; i++) out[i] = sum_sq_512(M + (size_t)i * dim, dim);
    } else {
        #pragma omp parallel for num_threads(n_threads) schedule(auto)
        for (int i = 0; i < rows; i++) out[i] = sum_sq_512(M + (size_t)i * dim, dim);
    }
}

inline float dot_product_512(const float* __restrict__ a,
                             const float* __restrict__ b, int dim) {
    __m512 acc0=_mm512_setzero_ps(), acc1=_mm512_setzero_ps(),
           acc2=_mm512_setzero_ps(), acc3=_mm512_setzero_ps();
    int d = 0;
    for (; d + 63 < dim; d += 64) {
        acc0=_mm512_fmadd_ps(_mm512_loadu_ps(a+d),    _mm512_loadu_ps(b+d),    acc0);
        acc1=_mm512_fmadd_ps(_mm512_loadu_ps(a+d+16), _mm512_loadu_ps(b+d+16), acc1);
        acc2=_mm512_fmadd_ps(_mm512_loadu_ps(a+d+32), _mm512_loadu_ps(b+d+32), acc2);
        acc3=_mm512_fmadd_ps(_mm512_loadu_ps(a+d+48), _mm512_loadu_ps(b+d+48), acc3);
    }
    for (; d + 15 < dim; d += 16)
        acc0=_mm512_fmadd_ps(_mm512_loadu_ps(a+d), _mm512_loadu_ps(b+d), acc0);
    __m512 s=_mm512_add_ps(_mm512_add_ps(acc0,acc1),_mm512_add_ps(acc2,acc3));
    float res=_mm512_reduce_add_ps(s);
    for (; d<dim; d++) res += a[d]*b[d];
    return res;
}

inline float sum_sq_diff_512(const float* __restrict__ a,
                              const float* __restrict__ b, int dim) {
    __m512 acc0=_mm512_setzero_ps(), acc1=_mm512_setzero_ps(),
           acc2=_mm512_setzero_ps(), acc3=_mm512_setzero_ps();
    int d = 0;
    for (; d + 63 < dim; d += 64) {
        __m512 e0=_mm512_sub_ps(_mm512_loadu_ps(a+d),    _mm512_loadu_ps(b+d)),
               e1=_mm512_sub_ps(_mm512_loadu_ps(a+d+16), _mm512_loadu_ps(b+d+16)),
               e2=_mm512_sub_ps(_mm512_loadu_ps(a+d+32), _mm512_loadu_ps(b+d+32)),
               e3=_mm512_sub_ps(_mm512_loadu_ps(a+d+48), _mm512_loadu_ps(b+d+48));
        acc0=_mm512_fmadd_ps(e0,e0,acc0); acc1=_mm512_fmadd_ps(e1,e1,acc1);
        acc2=_mm512_fmadd_ps(e2,e2,acc2); acc3=_mm512_fmadd_ps(e3,e3,acc3);
    }
    for (; d + 15 < dim; d += 16) {
        __m512 e=_mm512_sub_ps(_mm512_loadu_ps(a+d),_mm512_loadu_ps(b+d));
        acc0=_mm512_fmadd_ps(e,e,acc0);
    }
    __m512 s=_mm512_add_ps(_mm512_add_ps(acc0,acc1),_mm512_add_ps(acc2,acc3));
    float res=_mm512_reduce_add_ps(s);
    for (; d<dim; d++) { float diff=a[d]-b[d]; res+=diff*diff; }
    return res;
}

} 
#endif  
