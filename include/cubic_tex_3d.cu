#ifndef _CUBIC_TEX3D_CU_
#define _CUBIC_TEX3D_CU_

#include <cuda_runtime.h>
#include <helper_math.h>

#include "bspline_kernel.cu"


/// <summary>
/// Tricubic interpolated texture lookup function. using unnormalized coordinates.
/// Fast implementation, using 8 trilinear lookups.
/// </summary>
/// <typeparam name="T">Type of the texture data.</typeparam>
/// <param name="texObj">Texture object.</param>
/// <param name="coord">Unnormalized 3D texture coordinate.</param>
template<typename T>
__device__ float tricubic_tex3d(cudaTextureObject_t texObj, float3 coord)
{
	float3 index = floorf(coord);
	float3 fraction = coord - index;

	// 1) compute B-spline weights w0..w3 along each axis
	float3 w0, w1, w2, w3;
	bspline_weights(fraction, w0, w1, w2, w3);

	// 2) combine into two pairs per axis
	float3 g0 = w0 + w1;   // weight for the "lower" linear sample
	float3 g1 = w2 + w3;   // weight for the "upper" linear sample

	// 3) derive the two sampling positions h0, h1 per axis
	//    so that a single linear fetch combines w0/w1 (or w2/w3)
	float3 h0 = (w1 / g0) - 1.0f + index;
	float3 h1 = (w3 / g1) + 1.0f + index;

	// 4) perform 8 trilinear lookups, interleaving fetch+blend

	// --- z = h0.z slice ---
	float t000 = tex3D<T>(texObj, h0.x + 0.5f, h0.y + 0.5f, h0.z + 0.5f);
	float t100 = tex3D<T>(texObj, h1.x + 0.5f, h0.y + 0.5f, h0.z + 0.5f);
	t000 = g0.x * t000 + g1.x * t100;  // blend along x

	float t010 = tex3D<T>(texObj, h0.x + 0.5f, h1.y + 0.5f, h0.z + 0.5f);
	float t110 = tex3D<T>(texObj, h1.x + 0.5f, h1.y + 0.5f, h0.z + 0.5f);
	t010 = g0.x * t010 + g1.x * t110;  // blend along x

	float z0 = g0.y * t000 + g1.y * t010;  // blend along y

	// --- z = h1.z slice ---
	float t001 = tex3D<T>(texObj, h0.x + 0.5f, h0.y + 0.5f, h1.z + 0.5f);
	float t101 = tex3D<T>(texObj, h1.x + 0.5f, h0.y + 0.5f, h1.z + 0.5f);
	t001 = g0.x * t001 + g1.x * t101;

	float t011 = tex3D<T>(texObj, h0.x + 0.5f, h1.y + 0.5f, h1.z + 0.5f);
	float t111 = tex3D<T>(texObj, h1.x + 0.5f, h1.y + 0.5f, h1.z + 0.5f);
	t011 = g0.x * t011 + g1.x * t111;

	float z1 = g0.y * t001 + g1.y * t011;

	// 5) final blend along z
	return g0.z * z0 + g1.z * z1;
}

#endif // _CUBIC_TEX3D_CU_
