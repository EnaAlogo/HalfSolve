#include "vec.hpp"
#include "slice.hpp"
#include <execution>
#include "alloc.hpp"

namespace smath {

    template<typename T>
    std::enable_if_t<UseMixedType<T>::value ==0,void>
        AddRowMultiple(void* x, int64_t m, int64_t n, int64_t dst, int64_t source, T multiple) {
        using vec = Vec<T>;
        using vec_t = Vec_t<T>;
        using s_t = T; 
        using Simd = Simd<T>;
        constexpr int VEC_LEN = Simd::size(); 

        s_t* r1 = reinterpret_cast<s_t*>(x) + n * dst;
        s_t const* r2 = reinterpret_cast<s_t*>(x) + n * source;
        
        std::ptrdiff_t const vec_len = n - n % VEC_LEN; 

        for (size_t i = 0; i < n - vec_len; ++i)
        {
            r1[i] += r2[i] * multiple;
        }
        for(uint64_t idx = n - vec_len ; idx < n ; idx+=VEC_LEN){
            auto const l2 = loadu(r2 + idx);  
            auto const me = loadu(r1 + idx); 
            storeu(fmadd(l2, set1(multiple), me), r1+idx);
        }
    }

    template<typename T>
    std::enable_if_t<UseMixedType<T>::value == 1, void>
        AddRowMultiple(void* x, int64_t m, int64_t n, int64_t dst, int64_t source, T multiple) {
        using vec = Vec<T>;
        using vec_t = Vec_t<T>;
        using s_t = T;
        using Simd = Simd<T>;
        constexpr int VEC_LEN = Simd::size();

        s_t* r1 = reinterpret_cast<s_t*>(x) + n * dst;
        s_t const* r2 = reinterpret_cast<s_t*>(x) + n * source;

        std::ptrdiff_t const vec_len = n - n % VEC_LEN;

        for (size_t i = 0; i < n - vec_len; ++i)
        {
            r1[i] += r2[i] * multiple; 
        }
        for (uint64_t idx = n - vec_len; idx < n; idx += VEC_LEN) {
            auto const [other_l, other_h] = vec::loadu(r2 + idx);
            auto const [this_l, this_h] = vec::loadu(r1 + idx);
            auto const m = Vec<float>::set1(multiple);
            vec::storeu(r1 + idx,
                Vec<float>::fmadd(other_l, m, this_l),
                Vec<float>::fmadd(other_h, m, this_h) 
            );
        }

    }

    template<typename T>
    std::enable_if_t<UseMixedType<T>::value == 0, T> 
        Dot(T const* x , T const* y, int64_t n)  
    {
        const int simd_s = Simd<T>::size(); 
        const int simd_s2 = simd_s * 2; 
        int i, n8 = n - n % simd_s2; 
        Simd<T> vs1 = set1<T>(0), vs2 = set1<T>(0);
        T s;  
        for (i = 0; i < n8; i += simd_s2) { 

            auto const vx1 = loadu(&x[i]);
            auto const vx2 = loadu(&x[i + simd_s]);
            auto const vy1 = loadu(&y[i]);
            auto const vy2 = loadu(&y[i + simd_s]);

            vs1 = fmadd(vx1, vy1, vs1);
            vs2 = fmadd(vx2, vy2, vs2);
        }

        for (s = 0; i < n; ++i) s += x[i] * y[i]; 

        s += sum((vs1 + vs2).raw);
        
        return static_cast<T>(s);
    }
    template<typename T>
    std::enable_if_t<UseMixedType<T>::value == 1, float>
        Dot(T const* x, T const* y, int64_t n)
    {
        const int simd_s = Simd<T>::size(); 
        const int simd_s2 = simd_s * 2; 
        int i, n8 = n - n % simd_s2; 
        Vec<float>::type vs1 = Vec<float>::set1(0), vs2 = Vec<float>::set1(0);
        float s;
        for (i = 0; i < n8; i += simd_s2) {
            auto const [vx1_l, vx1_h] = Vec<T>::loadu(&x[i]);
            auto const [vx2_l, vx2_h] = Vec<T>::loadu(&x[i+simd_s]);
            auto const [vy1_l, vy1_h] = Vec<T>::loadu(&y[i]);
            auto const [vy2_l, vy2_h] = Vec<T>::loadu(&y[i+simd_s]);

            vs1 = Vec<float>::fmadd(vx1_l, vy1_l, vs1);  
            vs1 = Vec<float>::fmadd(vx1_h, vy1_h, vs1); 
            vs2 = Vec<float>::fmadd(vx2_l, vy2_l, vs2);  
            vs2 = Vec<float>::fmadd(vx2_h, vy2_h, vs2); 
        }
        for (s = 0; i < n; ++i) s += x[i] * y[i];
        
        s += sum(Vec<float>::add(vs1, vs2)); 
        return s;
    }

    //mixed
    template<typename T>
    std::enable_if_t<UseMixedType<T>::value == 1, float>
        Dot(T const* x, float const* y, int64_t n)
    {
        const int simd_s = Simd<T>::size();
        const int simd_s2 = simd_s * 2;
        int i, n8 = n - n % simd_s2;
        Vec<float>::type vs1 = Vec<float>::set1(0), vs2 = Vec<float>::set1(0);
        float s;
        for (i = 0; i < n8; i += simd_s2) {
            auto const [vx1_l, vx1_h] = Vec<T>::loadu(&x[i]);
            auto const [vx2_l, vx2_h] = Vec<T>::loadu(&x[i + simd_s]);

            auto const vy1_l = Vec<float>::loadu(&y[i]);
            auto const vy1_h = Vec<float>::loadu(&y[i + Simd<float>::size()]);
            auto const vy2_l = Vec<float>::loadu(&y[i + Simd<float>::size() *2]); 
            auto const vy2_h = Vec<float>::loadu(&y[i + Simd<float>::size() * 3]);

            vs1 = Vec<float>::fmadd(vx1_l, vy1_l, vs1);
            vs1 = Vec<float>::fmadd(vx1_h, vy1_h, vs1);
            vs2 = Vec<float>::fmadd(vx2_l, vy2_l, vs2);
            vs2 = Vec<float>::fmadd(vx2_h, vy2_h, vs2);
        }
        for (s = 0; i < n; ++i) s += static_cast<float>(x[i]) * y[i];

        s += sum(Vec<float>::add(vs1, vs2)); 
        return s;
    }

    template<typename T>
    std::enable_if_t<UseMixedType<T>::value == 0, void>
        Multiply(void* x, double num, int64_t size) {
        if (num == 0) {
            std::memset(x, 0, sizeof(T) * size);
        }
        using vec = Vec<T>;
        using vec_t = Vec_t<T>;
        using s_t = T;
        using Simd = Simd<T>;
        constexpr int VEC_LEN = Simd::size();

        std::ptrdiff_t const vec_len = size - size % VEC_LEN;

        for (size_t i = 0; i < size - vec_len; ++i)
        {
            *(reinterpret_cast<s_t*>(x)+i) *= num; 
        }

        for (uint64_t idx = size - vec_len; idx < size; idx += VEC_LEN) {
            vec_t l = vec::loadu(reinterpret_cast<s_t const*>(x) + idx);
            l = vec::mul(l, vec::set1(num));
            vec::storeu(reinterpret_cast<s_t*>(x) + idx, l);
        };

    }
    template<typename T>
    std::enable_if_t<UseMixedType<T>::value == 1, void>
        Multiply(void* x, double num, int64_t size) {
        if (num == 0) {
            std::memset(x, 0, sizeof(T) * size);
        }
        using vec = Vec<T>;
        using vec_t = Vec_t<T>;
        using s_t = T;
        using Simd = Simd<T>;
        constexpr int VEC_LEN = Simd::size();

        std::ptrdiff_t const vec_len = size - size % VEC_LEN;

        for (size_t i = 0; i < size - vec_len; ++i)
        {
            *(reinterpret_cast<s_t*>(x) + i) *= num;
        }

        for (uint64_t idx = size - vec_len; idx < size; idx += VEC_LEN) {
            auto[lo,hi] = vec::loadu(reinterpret_cast<s_t const*>(x) + idx);
            auto const mult = Vec<float>::set1(num);
            lo = Vec<float>::mul(lo, mult);
            hi = Vec<float>::mul(hi, mult);
            vec::storeu(reinterpret_cast<s_t*>(x) + idx, lo ,hi);
        }

    }

    template<typename T>
    void Copy(T* dst, T const* src, uint64_t numel) { 
        std::memcpy(dst, src, sizeof(T) * numel);
    }

    template<typename T>
    std::enable_if_t<UseMixedType<T>::value == 1, void> Copy(T* dst, float const* src, uint64_t size) {
        using vec = Vec<T>; 
        using vec_t = Vec_t<T>; 
        using Simd = Simd<T>; 
        constexpr int VEC_LEN = Simd::size();  

        std::ptrdiff_t const vec_len = size - size % VEC_LEN; 

        for (size_t i = 0; i < size - vec_len; ++i) 
        {
            dst[i] = src[i];
        }

        for (uint64_t idx = size - vec_len; idx < size; idx += VEC_LEN) {
            auto const lo = Vec<float>::loadu(&src[idx]); 
            auto const high = Vec<float>::loadu(&src[idx + smath::Simd<float>::size()]);  
            vec::storeu(dst + idx, lo, high);
        }
    }
    template<typename T>
    std::enable_if_t<UseMixedType<T>::value == 1, void> Copy(float* dst, T const* src, uint64_t size) {
        using vec = Vec<T>;
        using vec_t = Vec_t<T>;
        using Simd = Simd<T>;
        constexpr int VEC_LEN = Simd::size();

        std::ptrdiff_t const vec_len = size - size % VEC_LEN;

        for (size_t i = 0; i < size - vec_len; ++i)
        {
            dst[i] = src[i];
        }

        for (uint64_t idx = size - vec_len; idx < size; idx += VEC_LEN) {
            auto const [lo, hi] = vec::loadu(src + idx);
            Vec<float>::storeu(&dst[idx], lo);
            Vec<float>::storeu(&dst[idx + smath::Simd<float>::size()], hi);
        }
    }

    // (M N) X (N K) = (M K)
    template<typename T>
    AlignedMem<T> RowConcat(T const* x, T const* y, int64_t m, int64_t n, int64_t k)
    {
        auto temp = allocate<T>(m * (k + n));
        auto it = range_t(m);
        std::for_each(std::execution::par, it.begin(), it.end(), [&](int64_t idx) {
            T* dst = temp.get() + (idx * (k + n));
            std::memcpy(dst, x + idx * n, n * sizeof(T));
            std::memcpy(dst + n, y + idx * k, k * sizeof(T));
        });
        return temp;
    }

    template<typename T>
    std::enable_if_t<UseMixedType<T>::value==1,
    AlignedMem<float>> RowConcat(T const* x, float const* y, int64_t m, int64_t n, int64_t k)
    {
        auto temp = allocate<float>(m * (k + n));
        auto it = range_t(m);
        std::for_each(std::execution::par, it.begin(), it.end(), [&](int64_t idx) {
            float* dst = temp.get() + (idx * (k + n));
            Copy(dst, x + idx * n, n); 
            std::memcpy(dst+n, y + idx * k, k * sizeof(T)); 

            });
        return temp;
    }
    template<typename T>
    std::enable_if_t<UseMixedType<T>::value == 1,
        AlignedMem<float>> RowConcat(float const* x, T const* y, int64_t m, int64_t n, int64_t k)
    {
        auto temp = allocate<float>(m * (k + n));
        auto it = range_t(m);
        std::for_each(std::execution::par, it.begin(), it.end(), [&](int64_t idx) {
            float* dst = temp.get() + (idx * (k + n));
            std::memcpy(dst, x + idx * n, n * sizeof(float));
            Copy(dst+n, y + idx * k, k);
        });
        return temp;
    }



    template<typename T>
    void SwapRows(void* x, int64_t m, int64_t n, int64_t row1, int64_t row2) {
        if (row1 == row2) {
            return;
        }
        using vec = Vec<T>; 
        using vec_t = Vec_t<T>;
        constexpr size_t VEC_LEN = Simd<T>::size(); 

        T* r1 = reinterpret_cast<T*>(x) + n * row1;  
        T* r2 = reinterpret_cast<T*>(x) + n * row2;  
        std::ptrdiff_t const vec_len = n - n % VEC_LEN; 
        for (size_t i = 0; i < n - vec_len; ++i)
        {
            std::swap(r1[i], r2[i]);
        }
        for (uint64_t idx = n - vec_len; idx < n; idx += VEC_LEN) {
            __m128i const l = _mm_load_si128(reinterpret_cast<const __m128i*>(r1 + idx)); 
            __m128i const l2 = _mm_load_si128(reinterpret_cast<const __m128i*>(r2 + idx)); 
            _mm_storeu_si128(reinterpret_cast<__m128i*>(r2 + idx), l); 
            _mm_storeu_si128(reinterpret_cast<__m128i*>(r1 + idx), l2);
        }
    }

    template<typename T>
    void UpperTriangular(T* x, int64_t m, int64_t n) {
        int64_t j = 0, i = 0;
        while (i < m && j < n) {
            int64_t k = i;
            while (k < m && x[k * n + j] == T(0.f)) ++k;
            if (k < m) {
                smath::SwapRows<T>(x, m, n, i, k);
                for (k = i + 1; k < m; ++k) {
                    smath::AddRowMultiple<T>(x, m, n, k, i, -x[k * n + j] / x[i * n + i]);

                }
                ++i;
            }
            ++j;

        }
    }
    template<typename T>
    void GaussJordanElimination(T* x, int64_t m, int64_t n) {
        int64_t j = 0, i = 0;
        while (i < m && j < n) {
            int64_t k = i;
            while (k < m && x[k * n + j] == T(0.f)) ++k;
            if (k < m) {
                smath::SwapRows<T>(x, m, n, i, k);
                //mulitply row
                smath::Multiply<T>(x + i * n, T(1.f) / x[i * n + i], n);
                for (k = 0; k < m; ++k) {
                    if (k != i) {
                        smath::AddRowMultiple<T>(x, m, n, k, i, -x[k * n + j]);
                    }
                }
                ++i;
            }
            ++j;

        }
    }
}//end smath