#pragma once
#include <cstddef>
#include <string>
#include <stdexcept>
#include <memory>

namespace smath {

	template<typename T>
	struct Deleter {
		void operator()(T* block)const { 
#ifdef _MSC_VER 
			_aligned_free(block); 
#else
			free(block);
#endif

		}
	};

	template<typename T>
	using AlignedMem = std::unique_ptr<T[], Deleter<T>>;

	template<typename T>
	AlignedMem<T> allocate(size_t numel)
	{
		const static int gAlignment = 64;
		const size_t nbytes = sizeof(T) * numel;
		if (nbytes == 0) {
			return nullptr;
		}
		if (((ptrdiff_t)nbytes) < 0) {
			throw std::runtime_error(
				"cant malloc negative number of bytes: " + std::to_string(nbytes));
		}
		void* data;

#if defined(_MSC_VER)
		data = _aligned_malloc(nbytes, gAlignment);
		if (!data) {
			throw std::runtime_error(
				"not enough memory: tried to allocate "
				+ std::to_string(nbytes)
				+ std::string(" bytes."));
		}
#else
		int err = posix_memalign(&data, gAlignment, nbytes);
		if (!(err == 0)) {
			throw std::runtime_error(
				"tried to allocate "
				+ std::to_string(nbytes)
				+ std::string(" bytes. Error code ")
				+ std::to_string(err)
				+ std::string(" (")
				+ std::string(strerror(err))
				+ ")");
		}
#endif

		return { reinterpret_cast<T*>(data), Deleter<T>{} };

	}


}//end smath