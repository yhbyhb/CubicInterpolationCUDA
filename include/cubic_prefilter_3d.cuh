#ifndef CUBIC_PREFILTER_3D_CUH
#define CUBIC_PREFILTER_3D_CUH

extern "C" void prefilter_volume_cubic_b_spline(
    cudaSurfaceObject_t surf_volume,
    cudaExtent extent);

extern "C" void convert_to_device_array(
    const short* h_volume,
    cudaArray_t cuda_array,
    cudaExtent extent);

#endif  // CUBIC_PREFILTER_3D_CUH
