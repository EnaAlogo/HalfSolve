#pragma once
#include "vec.hpp"
#include "diagonals.hpp"
#include "linalg_helpers.hpp"
#include <random>
#include <optional>

namespace smath {

    template<typename T>
    struct Mat {
        Mat(int64_t m, std::optional<int64_t> n = {})  
            :m(m), n(n.value_or(m)), data_{ smath::allocate<T>(m * n.value_or(m)) } {}

        const T& at(int64_t i, int64_t j)const {
            return data_[i * n + j];
        }

        T& at(int64_t i, int64_t j) {
            return data_[i * n + j];
        }

        int64_t ncols()const { return n; }
        int64_t nrows()const { return m; }

        void reset(smath::AlignedMem<T> other) {
            data_ = std::move(other);
        }

        T* data() {
            return data_.get();
        }
        T const* data() const {
            return data_.get();
        }

    private:
        smath::AlignedMem<T> data_;
        int64_t m, n;
    };


    template<typename T>
    void FillRandn(T* ptr, int64_t size) {
        std::default_random_engine x;
        auto dist = std::normal_distribution();
        for (int i = 0; i < size; ++i) {
            ptr[i] = float(dist(x));
        }
    }

    template<typename T>
    Mat<T> Clone(Mat<T> const& mat) {
        auto out = Mat<T>(mat.nrows(), mat.ncols());
        std::memcpy(out.data(), mat.data(), sizeof(T) * mat.ncols() * mat.nrows());
        return out;
    }

    template<typename T>
    double Det(Mat<T> const& mat) {
        if (mat.nrows() != mat.ncols()) {
            throw std::runtime_error("Det() isnt defined for non squared matrices");
        }
        // det = product of the elements of the diagonal if the matrix is upper triangular
        auto temp = Clone(mat);
        //get the upper triangular matrix via row ops
        UpperTriangular(temp.data(), temp.nrows(), temp.ncols());
        return smath::DiagonalProd(temp.data(), temp.ncols());
    }

    template<typename T>
    Mat<T> Inv(Mat<T> const& mat) {
        if (mat.nrows() != mat.ncols()) {
            throw std::runtime_error("Inv() isnt defined for non squared matrices");
        }
        auto ident = Mat<T>(mat.nrows(), mat.ncols());
        // solve X X_inv = I 
        smath::FillIdentity(ident.data(), ident.nrows());
        //get the augmented matrix
        auto cat = smath::RowConcat(mat.data(), ident.data(), mat.nrows(), mat.ncols(), mat.ncols());
        //solve the system
        GaussJordanElimination(cat.get(), mat.ncols(), mat.nrows() * 2);
        //write back into ident and return it
        auto it = range_t(0, mat.ncols());
        std::for_each(std::execution::par, it.begin(), it.end(), [&](int64_t idx) {
            std::memcpy(ident.data() + idx * ident.ncols(),
            cat.get() + idx * (2 * ident.ncols()) + ident.ncols(),
            sizeof(T) * ident.ncols());
            });

        return ident;
    }

    // (M N) (N K) = (M K)
    template<typename T>
    Mat<T> Solve(Mat<T> const& A, Mat<T> const& B) {
        if (A.nrows() != A.ncols()) {
            throw std::runtime_error("A has to be a square matrix");
        }
        if (A.nrows() != B.nrows()) {
            throw std::runtime_error("A and B must have the same number of rows for AX=B to exist");
        }
        //create the augmented (M N+K) matrix
        auto cat = smath::RowConcat(A.data(), B.data(), A.nrows(), A.ncols(), B.ncols());
        GaussJordanElimination(cat.get(), A.nrows(), A.ncols() + B.ncols());
        //the solution is the last K columns of every row in the augmented matrix
        auto out = Mat<T>(A.nrows(), B.ncols());
        //write back and return it
        auto it = range_t(0, A.nrows());
        std::for_each(std::execution::par, it.begin(), it.end(), [&](int64_t idx) {
            std::memcpy(out.data() + idx * out.ncols(),
            cat.get() + idx * (A.ncols() + B.ncols()) + A.ncols(),
            sizeof(T) * out.ncols());
            });
        return out;
    }

    //mixed
    template<typename T>
    std::enable_if_t<smath::UseMixedType<T>::value == 1, Mat<float>>
        Solve(Mat<T> const& A, Mat<float> const& B) {
        if (A.nrows() != A.ncols()) {
            throw std::runtime_error("A has to be a square matrix");
        }
        if (A.nrows() != B.nrows()) {
            throw std::runtime_error("A and B must have the same number of rows for AX=B to exist");
        }
        //create the augmented (M N+K) matrix
        auto cat = smath::RowConcat(A.data(), B.data(), A.nrows(), A.ncols(), B.ncols());
        GaussJordanElimination(cat.get(), A.nrows(), A.ncols() + B.ncols());
        //the solution is the last K columns of every row in the augmented matrix
        auto out = Mat<float>(A.nrows(), B.ncols());
        //write back and return it
        auto it = range_t(0, A.nrows());
        std::for_each(std::execution::par, it.begin(), it.end(), [&](int64_t idx) {
            std::memcpy(out.data() + idx * out.ncols(),
            cat.get() + idx * (A.ncols() + B.ncols()) + A.ncols(),
            sizeof(float) * out.ncols());
            });
        return out;
    }
    template<typename T>
    std::enable_if_t<smath::UseMixedType<T>::value == 1, Mat<float>>
        Solve(Mat<float> const& A, Mat<T> const& B) {
        if (A.nrows() != A.ncols()) {
            throw std::runtime_error("A has to be a square matrix");
        }
        if (A.nrows() != B.nrows()) {
            throw std::runtime_error("A and B must have the same number of rows for AX=B to exist");
        }
        //create the augmented (M N+K) matrix
        auto cat = smath::RowConcat(A.data(), B.data(), A.nrows(), A.ncols(), B.ncols());
        GaussJordanElimination(cat.get(), A.nrows(), A.ncols() + B.ncols());
        //the solution is the last K columns of every row in the augmented matrix
        auto out = Mat<float>(A.nrows(), B.ncols());
        //write back and return it
        auto it = range_t(0, A.nrows());
        std::for_each(std::execution::par, it.begin(), it.end(), [&](int64_t idx) {
            std::memcpy(out.data() + idx * out.ncols(),
            cat.get() + idx * (A.ncols() + B.ncols()) + A.ncols(),
            sizeof(float) * out.ncols());
            });
        return out;
    }

}//end smath