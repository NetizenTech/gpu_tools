#include <cuda_stdint.h>

/* return hash of array - GPU|CPU */
__host__ __device__ uint64_t hash0(const uint8_t *s, const uint16_t N)
{
    uint64_t x = 0;

    for (uint16_t i = 0; i < N; i++)
    {
        x ^= s[i];
        x ^= (x >> 29) & 0x5555555555555555ULL;
        x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
        x ^= (x << 37) & 0xFFF7EEE000000000ULL;
        x ^= (x >> 43);
    }

    return x;
}

/* generate array of hashes from 2D array - GPU-global */
__global__ void hash0_kernel(const uint8_t *s_arr, uint64_t *h_arr, const uint32_t N, const uint16_t NN)
{
    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        h_arr[i] = hash0((uint8_t *)&s_arr[i * NN], NN);
}

/* search h indexes in array - GPU-global */
__global__ void search_kernel(const uint64_t *h_arr, const uint64_t h, uint32_t *r_arr, const uint32_t N,
                              const uint32_t NN)
{
    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        if (h_arr[i] == h)
        {
            const uint32_t idx = atomicAdd(&r_arr[0], 1) + 1;
            if (idx < NN)
                r_arr[idx] = i;
        }
}
