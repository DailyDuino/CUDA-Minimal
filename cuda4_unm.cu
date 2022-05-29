
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"





__global__ void copyImage_kernal( CudaImg sourceImage, CudaImg returnImage )
{
    // X,Y coordinates and check image dimensions
    //--- this remains the same ---//
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= t_color_cuda_img.m_size.y ) return;
    if ( l_x >= t_color_cuda_img.m_size.x ) return;
    //--- this remains the same ---//

    //----- Your Code here -----////


    uchar3 pixel = sourceImage.getpixelRGB(l_y,l_x);
    returnImage.getpixelRGB(l_y,l_x) = pixel;


}

void copyImage( CudaImg sourceImage, CudaImg returnImage )
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    
    dim3 l_blocks( ( sourceImage.m_size.x + l_block_size - 1 ) / l_block_size, ( sourceImage.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );


    // Calling kernal function below
    copyImage_kernal<<< l_blocks, l_threads >>>( sourceImage, returnImage ); 

    

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}






