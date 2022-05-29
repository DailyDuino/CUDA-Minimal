
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv {
}

// Function prototype from .cu file
void copyImage( CudaImg sourceImage, CudaImg returnImage );


int main( int t_numarg, char **t_arg )
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator( &allocator );

    if ( t_numarg < 2 )
    {
        printf( "Enter picture filename!\n" );
        return 1;
    }

    // Load image
    cv::Mat source_image = cv::imread( t_arg[ 1 ], cv::IMREAD_COLOR ); 

    if ( !source_image.data )
    {
        printf( "Unable to read file '%s'\n", t_arg[ 1 ] );
        return 1;
    }

    // create empty RGB image
    cv::Mat return_image( source_image.size(), CV_8UC3 );


    // convert cv::Mat Image to CudaImg
    CudaImg cuda_sourceImage, cuda_returnImage;

    cuda_sourceImage.cvtocudaRGB(source_image);
    cuda_returnImage.cvtocudaRGB(return_image);

    //call the function from .cu file
    copyImage(cuda_sourceImage,cuda_returnImage);

    

    // show images

    cv::imshow( "Source Image", source_image );
    cv::imshow( "Return Image", return_image );

    cv::waitKey( 0 );
}

