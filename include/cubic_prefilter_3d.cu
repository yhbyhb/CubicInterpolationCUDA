#ifndef _CUBIC_PREFILTER_3D_CU_
#define _CUBIC_PREFILTER_3D_CU_

#include "cubic_prefilter_kernel.cu"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>


//------------------------------------------------------------------------------
// Generic, axis-parametrized kernel

/// <summary>
/// Convert the volume to b-spline coefficients along the specified axis
/// </summary>
/// <param name="surfVolume">surface object for in-place processing</param>
/// <param name="width">width of the volume</param>
/// <param name="height">height of the volume</param>
/// <param name="depth">depth of the volume</param>
template<typename FloatN, int axis>
__global__ void convert_to_coeffs(
	cudaSurfaceObject_t surfVolume,
	uint32_t width,
	uint32_t height,
	uint32_t depth)
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
	int3 pos;
	uint32_t length;

	if constexpr (axis == 0) {
		pos = make_int3(0, i, j);
		length = width;
	}
	else if constexpr (axis == 1) {
		pos = make_int3(i, 0, j);
		length = height;
	}
	else /* axis == 2 */ {
		pos = make_int3(i, j, 0);
		length = depth;
	}

	convert_to_interpolation_coeffs_axis<FloatN, axis>(surfVolume, length, pos);
}

//--------------------------------------------------------------------------
// Exported functions
//--------------------------------------------------------------------------
/// <summary>
/// Prefilter the volume for cubic b-spline interpolation.
/// </summary>
/// <param name="surfVolume"></param>
/// <param name="extent"></param>
void prefilter_volume_cubic_b_spline(
	cudaSurfaceObject_t surfVolume,
	cudaExtent extent)
{
	uint32_t width = static_cast<uint32_t>(extent.width);
	uint32_t height = static_cast<uint32_t>(extent.height);
	uint32_t depth = static_cast<uint32_t>(extent.depth);

	// Try to determine the optimal block dimensions
	uint32_t dimX = min(min(get_power_of_two_divider(width), get_power_of_two_divider(height)), 64);
	uint32_t dimY = min(min(get_power_of_two_divider(depth), get_power_of_two_divider(height)), 512/dimX);
	dim3 dimBlock(dimX, dimY);

	// Replace the voxel values by the b-spline coefficients
	dim3 dimGridX(height / dimBlock.x, depth / dimBlock.y);
	convert_to_coeffs<float, 0><<<dimGridX, dimBlock>>>(surfVolume, width, height, depth);

	dim3 dimGridY(width / dimBlock.x, depth / dimBlock.y);
	convert_to_coeffs<float, 1><<<dimGridY, dimBlock>>>(surfVolume, width, height, depth);

	dim3 dimGridZ(width / dimBlock.x, height / dimBlock.y);
	convert_to_coeffs<float, 2><<<dimGridZ, dimBlock>>>(surfVolume, width, height, depth);
}

/// <summary>
/// Convert the T type host volume to float type device array.
/// </summary>
/// <param name="h_volume">volume in host memory</param>
/// <param name="cuda_array">cudaArray to copy the data into</param>
/// <param name="extent">extent of the volume</param>
template<typename T>
extern "C" void convert_to_device_array(
	const T* h_volume,
	cudaArray_t cuda_array,
	cudaExtent extent)
{
	const size_t samples_per_slice = extent.width * extent.height;

	// Allocate per‐slice Thrust buffers
	thrust::device_vector<T> d_in(samples_per_slice);
	thrust::device_vector<float> d_out(samples_per_slice);

	// Copy/transform each slice and memcpy into the cudaArray
	for (auto z = 0; z < extent.depth; ++z) {
		// copy host T → d_in
		thrust::copy(h_volume + z * samples_per_slice,
			h_volume + (z + 1) * samples_per_slice,
			d_in.begin());

		// transform T→floats into d_out
		thrust::transform(
			thrust::cuda::par,
			d_in.begin(), d_in.end(),
			d_out.begin(),
			[] __host__ __device__(T v) {
				return static_cast<float>(v);
			}
		);

		// set up a 2D srcPtr for this slice
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(
			(void*)thrust::raw_pointer_cast(d_out.data()),
			extent.width * sizeof(float),
			extent.width,
			extent.height
		);
		copyParams.srcPos = make_cudaPos(0, 0, 0);
		copyParams.dstArray = cuda_array;
		copyParams.extent = make_cudaExtent(
			extent.width,
			extent.height,
			1
		);
		copyParams.dstPos = make_cudaPos(0, 0, z);
		copyParams.kind = cudaMemcpyDeviceToDevice;

		// perform the copy (device→device)
		cudaMemcpy3D(&copyParams);
	}
}

#endif  //_CUBIC_PREFILTER_3D_CU_
