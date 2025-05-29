#ifndef _CUBIC_PREFILTER_KERNEL_CU_
#define _CUBIC_PREFILTER_KERNEL_CU_


inline __device__ __host__ uint32_t get_power_of_two_divider(uint32_t n)
{
	if (n == 0) return 0;
	uint32_t divider = 1;
	while ((n & divider) == 0) divider <<= 1;
	return divider;
}

// The code below is based on the work of Philippe Thevenaz.
// See <http://bigwww.epfl.ch/thevenaz/interpolation/>

#define POLE (sqrt(3.0f)-2.0f)  //pole for cubic b-spline

//------------------------------------------------------------------------------
// Helpers to read from and write to 3D surface at pos
template<typename FloatN>
__device__ inline FloatN surfRead(cudaSurfaceObject_t surf, const int3& pos) {
	return surf3Dread<FloatN>(surf,
		pos.x * sizeof(FloatN),
		pos.y,
		pos.z);
}

template<typename FloatN>
__device__ inline void surfWrite(cudaSurfaceObject_t surf, const FloatN& val, const int3& pos) {
	surf3Dwrite<FloatN>(val,
		surf,
		pos.x * sizeof(FloatN),
		pos.y,
		pos.z);
}

//------------------------------------------------------------------------------
// Generic, axis-parametrized causal initialization
template<typename FloatN, int axis>
__device__ FloatN init_causal_coefficient_axis(
	cudaSurfaceObject_t surfVolume,
	uint32_t length,
	int3 pos)
{
	const uint32_t horizon = min(12u, length);
	float zn = POLE;
	FloatN sum = surfRead<FloatN>(surfVolume, pos);

	int* coord = nullptr;
	if constexpr (axis == 0) coord = &pos.x;
	if constexpr (axis == 1) coord = &pos.y;
	if constexpr (axis == 2) coord = &pos.z;

	for (uint32_t n = 0; n < horizon; ++n) {
		(*coord)++;
		sum += zn * surfRead<FloatN>(surfVolume, pos);
		zn *= POLE;
	}
	return sum;
}

//------------------------------------------------------------------------------
// anticausal initialization
template<typename FloatN>
__device__ FloatN init_anticausal_coefficient(
	FloatN value)
{
	return((POLE / (POLE - 1.0f)) * value);
}

//------------------------------------------------------------------------------
/// <summary>
/// axis-parametrized conversion to interpolation coefficients
/// </summary>
/// <param name="surfVolume">surface object for in-place processing</param>
/// <param name="length">number of samples or coefficients</param>
/// <param name="pos">position in the volume</param>
template<typename FloatN, int axis>
__device__ void convert_to_interpolation_coeffs_axis(
	cudaSurfaceObject_t surfVolume,
	uint32_t length,
	int3 pos)
{
	// overall gain
	const float lambda = (1.0f - POLE) * (1.0f - 1.0f / POLE);

	// pointer to the chosen coordinate
	int* coord = nullptr;
	if constexpr (axis == 0) coord = &pos.x;
	if constexpr (axis == 1) coord = &pos.y;
	if constexpr (axis == 2) coord = &pos.z;

	// causal initialization + write
	FloatN previous_c = lambda * init_causal_coefficient_axis<FloatN, axis>(surfVolume, length, pos);
	surfWrite<FloatN>(surfVolume, previous_c, pos);

	// causal recursion
	for (uint32_t n = 1; n < length; ++n) {
		(*coord)++;
		FloatN sample = surfRead<FloatN>(surfVolume, pos);
		previous_c = lambda * sample + POLE * previous_c;
		surfWrite<FloatN>(surfVolume, previous_c, pos);
	}

	// anticausal initialization + write
	FloatN last_sample = surfRead<FloatN>(surfVolume, pos);
	previous_c = init_anticausal_coefficient(last_sample);
	surfWrite<FloatN>(surfVolume, previous_c, pos);

	// anticausal recursion
	for (int n = int(length) - 2; n >= 0; --n) {
		(*coord)--;
		FloatN sample = surfRead<FloatN>(surfVolume, pos);
		previous_c = POLE * (previous_c - sample);
		surfWrite<FloatN>(surfVolume, previous_c, pos);
	}
}

#endif // _CUBIC_PREFILTER_KERNEL_CU_
