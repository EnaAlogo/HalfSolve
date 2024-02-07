#pragma once


//this is how DirectXMath does this
#if defined(_MSC_VER) && !defined(_M_ARM) && !defined(_M_ARM64) && !defined(_M_HYBRID_X86_ARM64) && !defined(_M_ARM64EC) && (!_MANAGED) && (!_M_CEE) && (!defined(_M_IX86_FP) || (_M_IX86_FP > 1)) && !defined(_XM_NO_INTRINSICS_) && !defined(_XM_VECTORCALL_)
#define VECTORCALL __vectorcall
#elif defined(__GNUC__)
#define VECTORCALL
#else
#define VECTORCALL __fastcall
#endif



namespace smath {

    template<typename T>
    struct UseMixedType : std::false_type {}; 


}//end smath