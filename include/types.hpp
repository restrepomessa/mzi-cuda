#pragma once
#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>

typedef std::complex<float> ComplexCPU;
typedef cuFloatComplex ComplexGPU;