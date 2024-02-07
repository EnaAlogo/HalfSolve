#include "common.hpp"
#include <execution>
#include <algorithm>

namespace smath {

    template<typename T>
    struct DiagonalIteratorImpl {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = value_type*;  
        using reference = value_type&;
        using const_reference = value_type const&;

        constexpr DiagonalIteratorImpl()
            :x(nullptr), m(0), i(0) {}

        constexpr DiagonalIteratorImpl(DiagonalIteratorImpl const& o)
            : x(o.x), m(o.m), i(o.i) {}

        constexpr DiagonalIteratorImpl(T* ptr, int64_t m, int64_t i = 0)
            : x(ptr), m(m), i(i) {}

        constexpr DiagonalIteratorImpl& operator=(const DiagonalIteratorImpl& other) {
            if (this != &other) {
                x = other.x;
                m = other.m;
                i = other.i;
            }
            return *this;
        }

        constexpr reference operator*()  { return x[i * m + i]; }
        constexpr const_reference operator*() const{ return x[i * m + i]; }

        constexpr pointer operator->() { return x; }

        constexpr reference operator[](int64_t i) {
            return x[i * m + i];
        }

        constexpr const_reference operator[](int64_t i)const {
            return x[i * m + i];
        }


        constexpr DiagonalIteratorImpl& operator++() {
            i++;
            return *this;
        }
        constexpr DiagonalIteratorImpl operator ++(int) {
            auto it = *this;
            ++(*this);
            return it;
        }
        constexpr friend bool operator== (const DiagonalIteratorImpl& a, const DiagonalIteratorImpl& b) { return a.i == b.i; };
        constexpr friend bool operator!= (const DiagonalIteratorImpl& a, const DiagonalIteratorImpl b) { return a.i != b.i; };
    private:
        T* x;
        int64_t m;
        int64_t i;
    };

    //MxM
    template<typename T>
    struct DiagonalIterator {
        constexpr DiagonalIterator(T* ptr, int64_t m)
            :ptr(ptr), m(m) {}

        constexpr DiagonalIteratorImpl<T> begin()const {
            return DiagonalIteratorImpl(ptr, m, 0);
        }

        constexpr DiagonalIteratorImpl<T> end()const {
            return DiagonalIteratorImpl(ptr, m, m);
        }
    private:
        T* ptr;
        int64_t m;
    };

    
    //MxM
    template<typename T>
    typename std::enable_if_t<smath::UseMixedType<T>::value == 0, T> DiagonalProd(T const* x, int64_t m) {
        auto it = DiagonalIterator(x, m);
        return std::reduce(it.begin(), it.end(), T(1.f), std::multiplies<T>{});
    }
    //MxM
    template<typename T>
    typename std::enable_if_t<smath::UseMixedType<T>::value == 1, float> DiagonalProd(T const* x, int64_t m) {
        auto it = DiagonalIterator(x, m);
        return std::reduce(it.begin(), it.end(), float(1.f), std::multiplies<T>{});
    }

    template<typename T>
    void FillIdentity(T* x, int64_t m) {
        std::memset(x, 0, sizeof(T) * m * m);
        for (T& val : DiagonalIterator(x, m)) {
            val = T(1.f);
        }
    }

}//end smath