#ifndef _BSPLINE_KERNEL_CU_
#define _BSPLINE_KERNEL_CU_


/// <summary>
/// Inline calculation of the bspline convolution weights, without conditional statements 
/// </summary>
template<typename T> inline __device__ void bspline_weights(T fraction, T& w0, T& w1, T& w2, T& w3)
{
	const T one_frac = 1.0f - fraction;
	const T squared = fraction * fraction;
	const T one_sqd = one_frac * one_frac;

	w0 = 1.0f / 6.0f * one_sqd * one_frac;
	w1 = 2.0f / 3.0f - 0.5f * squared * (2.0f - fraction);
	w2 = 2.0f / 3.0f - 0.5f * one_sqd * (2.0f - one_frac);
	w3 = 1.0f / 6.0f * squared * fraction;
}

#endif // _BSPLINE_KERNEL_CU_
