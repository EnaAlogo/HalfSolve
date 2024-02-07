#include <iostream>
#include <iomanip>

#include <matrix.hpp>



template<typename T>
smath::Mat<double> mm(smath::Mat<T> const& x, smath::Mat<T> const& y) {
	// (M N) (N K) = (M K)
	auto out = smath::Mat<double>(x.nrows(), y.ncols());
	for (size_t i = 0; i < x.nrows(); ++i) {
		for (size_t j = 0; j < y.ncols(); ++j) {
			out.at(i, j) = 0;
			for (size_t k = 0; k < x.ncols(); ++k) {
				out.at(i, j) += static_cast<float>(x.at(i, k)) * static_cast<float>(y.at(k, j));
			}
		}
	}
	return out;
}
template<typename T>
void Print(smath::Mat<T> const& x) {
	for (size_t i = 0; i < x.nrows(); ++i) {
		for (size_t j = 0; j < x.ncols(); ++j) {
			std::cout << x.at(i, j) <<" ";
		}
		std::cout << "\n";
	}
}

template<typename T>
void FillMatRandn(smath::Mat<T>& x) {
	smath::FillRandn(x.data(), x.ncols() * x.nrows());
}

template<typename T>
void RunInverse(int64_t problem_size) {
	auto x = smath::Mat<T>(problem_size);
	FillMatRandn(x);
	auto inv = smath::Inv(x);
	std::cout << "inverse :" << "\n";
	Print(inv);
	std::cout << "\n";
	//validate that X X_Inv = I (it wont be cuz of errors but its close)
	auto I = mm(x, inv);
	std::cout << "X*X_INV :" << "\n";
	Print(I);
	std::cout << "\n";
}
template<typename T>
void RunDet(int64_t problem_size) {
	auto x = smath::Mat<T>(problem_size);
	FillMatRandn(x);
	std::cout << "determinant : "<< smath::Det(x);
	std::cout << "\n";
}

void RunDot(int64_t problem_size) {
	auto x = smath::Mat<float>(1, problem_size);
	auto y = smath::Mat<float>(1, problem_size);
	FillMatRandn(x);
	FillMatRandn(y);

	std::cout << "single precision dot : ";
	std::cout << smath::Dot(x.data(), y.data(), problem_size)<<"\n";
	
	//mixed precision

	std::cout << "mixed precision dots : \n" ;
	auto z = smath::Mat<smath::bfloat16_t>(1, problem_size);
	smath::Copy(z.data(), x.data(), problem_size);
	
	std::cout << smath::Dot(z.data(), y.data(), problem_size) << "\n";

	auto f = smath::Mat<smath::float16_t>(1, problem_size);
	smath::Copy(f.data(), x.data(), problem_size);

	std::cout << smath::Dot(f.data(), y.data(), problem_size) <<"\n";

	//half precision 

	std::cout << "bfloat16 dot : " ;
	auto k = smath::Mat<smath::bfloat16_t>(1, problem_size);
	smath::Copy(k.data(), y.data(), problem_size);
	std::cout << smath::Dot(k.data(), z.data(), problem_size)<<"\n";


	std::cout << "half dot : " ;
	auto d = smath::Mat<smath::float16_t>(1, problem_size);
	smath::Copy(d.data(), y.data(), problem_size);
	std::cout << smath::Dot(d.data(), f.data(), problem_size)<<"\n";
}

template<typename T>
void RunSolve(int64_t asizes, int64_t bcol) {
	auto x = smath::Mat<T>(asizes);
	auto y = smath::Mat<T>(asizes, bcol);

	FillMatRandn(x);
	FillMatRandn(y);
	smath::Multiply<T>(x.data(), .4, asizes * asizes);
	smath::Multiply<T>(y.data(), -2, asizes * bcol);

	auto out = smath::Solve(x, y);
	std::cout << "A X = B : \n";
	Print(out);

	//validate that A X = B
	std::cout << "A X : \n";
	Print(mm(x, out));

	std::cout << "A B : \n";
	std::cout << "\n";
	Print(y);
	std::cout << "\n";
}

template<typename T>
std::enable_if_t<smath::UseMixedType<T>::value==1,void>
RunMixedSolve(int64_t asizes, int64_t bcol) {
	auto x = smath::Mat<float>(asizes);
	auto y = smath::Mat<T>(asizes, bcol); 

	FillMatRandn(x); 
	FillMatRandn(y); 

	auto out = smath::Solve(x, y);
	std::cout << "A X = B (mixed) : \n";
	Print(out);
	std::cout << "\n";

	//validate that A X = B
	std::cout << "A X : \n";
	Print(mm(x, out));

	std::cout << "B : \n";
	std::cout << "\n";
	Print(y);
	std::cout << "\n";
}
int main()
{
	std::cout << std::setprecision(5) << std::fixed;

	RunInverse<smath::float16_t>(4);
	RunInverse<float>(4);
	RunInverse<double>(4);

	RunDet<smath::bfloat16_t>(36);
	RunDet<float>(36);
	RunDet<double>(36);

	RunDot(100000);

	RunSolve<float>(5, 3);

	RunMixedSolve<smath::float16_t>(5, 1);
	RunMixedSolve<smath::bfloat16_t>(5, 1);

}
