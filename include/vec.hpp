#pragma once
#include "cvt.hpp"
#include "scalar_cvt.hpp"

namespace smath {
    template<typename T>
    struct Vec {
        using scalar_type = T;  
        static constexpr auto size = 128/(sizeof(T)*8); 
        using type = __m128i;

        static type loadu(void const* mem) { 
            return _mm_loadu_si128(reinterpret_cast<__m128i const*>(mem));
        }

        static type load(void const* mem) {
            return _mm_load_si128(reinterpret_cast<__m128i const*>(mem));
        }
        static void VECTORCALL store(void* mem, type x) {
            _mm_store_si128(x, reinterpret_cast<__m128i*>(mem));
        }

        static void VECTORCALL storeu(void* mem, type x) {  
            _mm_storeu_si128(x, reinterpret_cast<__m128i*>(mem)); 
        }
        static void VECTORCALL stream(void* mem, type x) {
            _mm_stream_si128(reinterpret_cast<__m128i*>(mem), x);
        }
    };

    template<>
    struct Vec<float16_t> {
        using scalar_type = float16_t;
        static constexpr auto size = 8;
        using type = __m128i;

        using compute_t = __m128;

        static std::tuple<compute_t, compute_t> loadu(void const* mem) {
            return bits::FloatToHalf<float16_t>::loadu(mem);
        }

        static void VECTORCALL storeu(void* mem, compute_t low, compute_t hi) {
            return bits::FloatToHalf<float16_t>::storeu(low, hi, mem);
        }

        static void VECTORCALL stream(void* mem, compute_t low, compute_t hi) {
            return bits::FloatToHalf<float16_t>::stream(low, hi, mem);
        }
    };

    template<>
    struct Vec<bfloat16_t> {
        using scalar_type = bfloat16_t;
        static constexpr auto size = 8;
        using type = __m128i;

        using compute_t = __m128;

        static std::tuple<compute_t, compute_t> loadu(void const* mem) {
            return bits::FloatToHalf<bfloat16_t>::loadu(mem);
        }

        static void VECTORCALL storeu(void* mem, compute_t low, compute_t hi) {
            return bits::FloatToHalf<bfloat16_t>::storeu(low, hi, mem);
        }
        static void VECTORCALL stream(void* mem, compute_t low, compute_t hi) {
            return bits::FloatToHalf<bfloat16_t>::stream(low, hi, mem);
        }
    };

    template<>
    struct Vec<float> {
        using scalar_type = float;
        static constexpr auto size = 4;
        using type = __m128;


        static type load(float const* mem) {
            return _mm_load_ps(mem);
        }

        static type loadu(float const* mem) {
            return _mm_loadu_ps(mem);
        }

        static void VECTORCALL store(float* mem, type x) {
            return _mm_store_ps(mem, x);
        }

        static void VECTORCALL storeu(float* mem, type x) {
            return _mm_storeu_ps(mem, x);
        }

        static void VECTORCALL stream(float* mem, type x) {
            return _mm_stream_ps(mem, x);
        }

        static type VECTORCALL fmadd(type x, type y, type z) {
            return _mm_fmadd_ps(x, y, z);
        }

        static type VECTORCALL add(type x, type z) {
            return _mm_add_ps(x, z);
        }

        static type VECTORCALL mul(type x, type z) {
            return _mm_mul_ps(x, z);
        }

        static type VECTORCALL sub(type x, type z) {
            return _mm_sub_ps(x, z);
        }

        static type set1(float x) {
            return _mm_set1_ps(x);
        }

        static type VECTORCALL div(type x, type z) {
            return _mm_div_ps(x, z);
        }

    };

    template<>
    struct Vec<double> {
        using scalar_type = double;
        static constexpr auto size = 2;
        using type = __m128d;
        
        static type load(double const* mem) {
            return _mm_loadu_pd(mem);
        }

        static void VECTORCALL store(double* mem, type x) {
            return _mm_storeu_pd(mem, x);
        }

        static type loadu(double const* mem) {
            return _mm_loadu_pd(mem);
        }

        static void VECTORCALL storeu(double* mem, type x) {
            return _mm_storeu_pd(mem, x);
        }

        static void VECTORCALL stream(double* mem, type x) {
            return _mm_stream_pd(mem, x);
        }

        static type VECTORCALL fmadd(type x, type y, type z) {
            return _mm_fmadd_pd(x, y, z);
        }

        static type VECTORCALL add(type x, type z) {
            return _mm_add_pd(x, z);
        }

        static type VECTORCALL mul(type x, type z) {
            return _mm_mul_pd(x, z);
        }

        static type VECTORCALL sub(type x, type z) {
            return _mm_sub_pd(x, z);
        }

        static type set1(double x) {
            return _mm_set1_pd(x);
        }

        static type VECTORCALL div(type x, type z) {
            return _mm_div_pd(x, z);
        }
    };
    
    template<typename T>
    using Vec_t = typename Vec<T>::type;


    template<typename T>
    struct Simd {
        using vec_type = Vec<T>;

        static constexpr size_t size() { return vec_type::size; }

        Vec_t<T> raw;
    };

    template<typename T,
        typename = std::enable_if_t< std::is_same_v<T, float> || std::is_same_v<T, double>>>
    inline Simd<T> VECTORCALL fmadd(Simd<T> x, Simd<T> y, Simd<T> z) {
        return Simd<T>(Vec<T>::fmadd(x.raw, y.raw, z.raw));
    }
    template<typename T>
    inline Simd<T> set1(T val) { 
        return Simd<T>(Vec<T>::set1(val));
    }

    template<typename T>
    inline Simd<T> loadu(T const* mem) {
        return Simd<T>(Vec<T>::loadu(mem));
    }
    template<typename T>
    inline void VECTORCALL storeu(Simd<T> x, T* mem) {
        Vec<T>::storeu(mem, x.raw);
    }
    template<typename T>
    inline void VECTORCALL stream(Simd<T> x, T* mem) {
        Vec<T>::stream(mem, x.raw);
    }

    template<typename T,
    typename = std::enable_if_t< std::is_same_v<T,float> || std::is_same_v<T,double>>>
    inline Simd<T> VECTORCALL operator +(Simd<T> x, Simd<T> y) {
        return Simd<T>(Vec<T>::add(x.raw, y.raw));
    }
    template<typename T,
        typename = std::enable_if_t< std::is_same_v<T, float> || std::is_same_v<T, double>>>
    inline Simd<T> VECTORCALL operator -(Simd<T> x, Simd<T> y) {
        return Simd<T>(Vec<T>::sub(x.raw, y.raw));
    }
    template<typename T,
        typename = std::enable_if_t< std::is_same_v<T, float> || std::is_same_v<T, double>>>
    inline Simd<T> VECTORCALL operator *(Simd<T> x, Simd<T> y) {
        return Simd<T>(Vec<T>::mul(x.raw, y.raw));
    }
    template<typename T,
        typename = std::enable_if_t< std::is_same_v<T, float> || std::is_same_v<T, double>>>
    inline Simd<T> VECTORCALL operator /(Simd<T> x, Simd<T> y) {
        return Simd<T>(Vec<T>::div(x.raw, y.raw));
    }

    template<typename T,
        typename = std::enable_if_t< std::is_same_v<T, float> || std::is_same_v<T, double>>>
    inline Simd<T>& VECTORCALL operator +=(Simd<T>& x, Simd<T> y) {
        x = x + y;
        return x;
    }
    template<typename T,
        typename = std::enable_if_t< std::is_same_v<T, float> || std::is_same_v<T, double>>>
    inline Simd<T>& VECTORCALL operator -=(Simd<T>& x, Simd<T> y) {
        x = x - y;
        return x;
    }
    template<typename T,
        typename = std::enable_if_t< std::is_same_v<T, float> || std::is_same_v<T, double>>>
    inline Simd<T>& VECTORCALL operator *=(Simd<T>& x, Simd<T> y) {
        x = x * y;
        return x;
    }

    template<typename T,
        typename = std::enable_if_t< std::is_same_v<T, float> || std::is_same_v<T, double>>>
    inline Simd<T>& VECTORCALL operator /=(Simd<T>& x, Simd<T> y) {
        x = x / y;
        return x;
    }

    template <typename T>
    inline __m128i load_n(const Vec_t<T>* p,size_t max_lanes_to_load) {
        __m128i const zeroed = _mm_setzero_si128();
        const size_t N = Simd<T>::size();
        const size_t num_of_lanes_to_load = std::min(max_lanes_to_load, N);
        std::memcpy(&zeroed,p->raw, num_of_lanes_to_load * sizeof(zeroed));
        return zeroed;
    }

    /*
    * hadd is SSE3
    */
    inline float VECTORCALL sum(__m128 v) {
        __m128 shuf = _mm_movehdup_ps(v);    // Duplicate odd elements 
        __m128 sums = _mm_add_ps(v, shuf);   // Add even and odd elements 
        shuf = _mm_movehl_ps(shuf, sums);    // High half -> low half 
        sums = _mm_add_ss(sums, shuf);       // Add low and high halves 
        return _mm_cvtss_f32(sums);          // Convert the result to a scala r float
    }

    inline double VECTORCALL sum(__m128d v) {
        __m128d shuf = _mm_shuffle_pd(v, v, 1);    // Swap the two doubles
        __m128d sums = _mm_add_pd(v, shuf);       // Add the doubles together
        return _mm_cvtsd_f64(sums);               // Convert the result to a scalar double
    }

}//end smath