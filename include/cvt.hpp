#pragma once
#include <immintrin.h>
#include <tuple>
#include "common.hpp"
#include "scalar_cvt.hpp"

namespace smath::bits {

    template<typename T>
    struct FloatToHalf {
    };

    template<>
    struct FloatToHalf<bfloat16_t>{ 

        static void VECTORCALL cvt_bp_ps(__m128i x, __m128* high, __m128* low) {
            __m128i const b2 = _mm_unpacklo_epi16(x, x);
            __m128i const b1 = _mm_unpackhi_epi16(x, x);
            *high = _mm_castsi128_ps(_mm_slli_epi32(b1, 16));
            *low = _mm_castsi128_ps(_mm_slli_epi32(b2, 16));
        }

        static __m128i VECTORCALL cvt_ps_bp(__m128 lo, __m128 hi) {
            __m128i const b1 = _mm_srai_epi32(_mm_castps_si128(lo), 16);
            __m128i const b2 = _mm_srai_epi32(_mm_castps_si128(hi), 16);
            return _mm_packs_epi32(b1, b2);
        }

        //returns (low,high)
        static inline std::tuple<__m128, __m128> VECTORCALL loadu(void const* mem) {
            __m128i const r = _mm_loadu_si128(reinterpret_cast<__m128i const*>(mem));
            std::tuple<__m128, __m128> out;
            cvt_bp_ps(r, &std::get<1>(out), &std::get<0>(out));
            return out;
        }

        static inline void VECTORCALL stream(__m128 low, __m128 high, void* mem) {
            _mm_stream_si128(reinterpret_cast<__m128i*>(mem), cvt_ps_bp(low , high));
        }
        static inline void VECTORCALL storeu(__m128 low, __m128 high, void* mem) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(mem), cvt_ps_bp(low , high));
        }
    };

    template<>
    struct FloatToHalf<float16_t> { 

        static __m128i VECTORCALL cvt_ps_ph(const __m128& f)
        {
            __m128  const mask_sign = _mm_set1_ps(-0.0f); 
            __m128i const c_f16max = _mm_set1_epi32((127 + 16) << 23); 
            __m128i const c_nanbit = _mm_set1_epi32(0x200);
            __m128i const c_nanlobits = _mm_set1_epi32(0x1ff);
            __m128i const c_infty_as_fp16 = _mm_set1_epi32(0x7c00);
            __m128i const c_min_normal = _mm_set1_epi32((127 - 14) << 23); 
            __m128i const c_subnorm_magic = _mm_set1_epi32(((127 - 15) + (23 - 10) + 1) << 23);
            __m128i const c_normal_bias = _mm_set1_epi32(0xfff - ((127 - 15) << 23)); 
            __m128  const justsign = _mm_and_ps(f, mask_sign);
            __m128  const absf = _mm_andnot_ps(mask_sign, f); 
            __m128i const absf_int = _mm_castps_si128(absf);
            __m128  const b_isnan = _mm_cmpunord_ps(absf, absf);
            __m128i const b_isregular = _mm_cmpgt_epi32(c_f16max, absf_int);
            __m128i const nan_payload = _mm_and_si128(_mm_srli_epi32(absf_int, 13), c_nanlobits); 
            __m128i const nan_quiet = _mm_or_si128(nan_payload, c_nanbit); 
            __m128i const nanfinal = _mm_and_si128(_mm_castps_si128(b_isnan), nan_quiet);
            __m128i const inf_or_nan = _mm_or_si128(nanfinal, c_infty_as_fp16); 
            __m128i const b_issub = _mm_cmpgt_epi32(c_min_normal, absf_int);
            __m128  const subnorm1 = _mm_add_ps(absf, _mm_castsi128_ps(c_subnorm_magic)); 
            __m128i const subnorm2 = _mm_sub_epi32(_mm_castps_si128(subnorm1), c_subnorm_magic); 
            __m128i const mantoddbit = _mm_slli_epi32(absf_int, 31 - 13); 
            __m128i const mantodd = _mm_srai_epi32(mantoddbit, 31);
            __m128i const round1 = _mm_add_epi32(absf_int, c_normal_bias);
            __m128i const round2 = _mm_sub_epi32(round1, mantodd);
            __m128i const normal = _mm_srli_epi32(round2, 13); 
            __m128i const nonspecial = _mm_or_si128(_mm_and_si128(subnorm2, b_issub), _mm_andnot_si128(b_issub, normal));
            __m128i const joined = _mm_or_si128(_mm_and_si128(nonspecial, b_isregular), _mm_andnot_si128(b_isregular, inf_or_nan));
            __m128i const sign_shift = _mm_srai_epi32(_mm_castps_si128(justsign), 16);
            __m128i const result = _mm_or_si128(joined, sign_shift);

            return result;
        }
        static void VECTORCALL cvt_ph_ps(__m128i src, __m128* high, __m128* low) {
            __m128i const mask_s8 = _mm_set1_epi16((short)0x8000);
            __m128i const mask_m8 = _mm_set1_epi16((short)0x03FF);
            __m128i const mask_e8 = _mm_set1_epi16((short)0x7C00);
            __m128i const bias_e4 = _mm_set1_epi32(0x0001C000);
            __m128i const s8 = _mm_and_si128(src, mask_s8);
            __m128i const m8 = _mm_and_si128(src, mask_m8);
            __m128i const e8 = _mm_and_si128(src, mask_e8);

            __m128i  s4a = _mm_unpacklo_epi16(s8, _mm_setzero_si128());
            s4a = _mm_slli_epi32(s4a, 16);

            __m128i m4a = _mm_unpacklo_epi16(m8, _mm_setzero_si128());
            m4a = _mm_slli_epi32(m4a, 13);

            __m128i e4a = _mm_unpacklo_epi16(e8, _mm_setzero_si128());
            e4a = _mm_add_epi32(bias_e4, e4a);
            e4a = _mm_slli_epi32(e4a, 13);

            *low = _mm_castsi128_ps(_mm_or_si128(s4a, _mm_or_si128(e4a, m4a)));

            //filling high
            __m128i s4b = _mm_unpackhi_epi16(s8, _mm_setzero_si128());
            s4b = _mm_slli_epi32(s4b, 16);

            __m128i m4b = _mm_unpackhi_epi16(m8, _mm_setzero_si128());
            m4b = _mm_slli_epi32(m4b, 13);

            __m128i e4b = _mm_unpackhi_epi16(e8, _mm_setzero_si128());
            e4b = _mm_add_epi32(bias_e4, e4b);
            e4b = _mm_slli_epi32(e4b, 13);

            *high = _mm_castsi128_ps(_mm_or_si128(s4b, _mm_or_si128(e4b, m4b)));
        }


        static void VECTORCALL storeu(__m128 low, __m128 high, void* mem) {
            __m128i const b1 = cvt_ps_ph(low);
            __m128i const b2 = cvt_ps_ph(high);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(mem), _mm_packs_epi32(b1, b2));
        }

        static void VECTORCALL stream(__m128 low, __m128 high, void* mem) {
            __m128i const b1 = cvt_ps_ph(low);
            __m128i const b2 = cvt_ps_ph(high);
            _mm_stream_si128(reinterpret_cast<__m128i*>(mem), _mm_packs_epi32(b1, b2));
        }

        //returns (low,high)
        static std::tuple<__m128, __m128> VECTORCALL loadu(void const* mem) {
            __m128i const r = _mm_loadu_si128(reinterpret_cast<__m128i const*>(mem));
            std::tuple<__m128, __m128> out;
            cvt_ph_ps(r, &std::get<1>(out), &std::get<0>(out));
            return out;
        }
    };



   

  

}//end megu::bits