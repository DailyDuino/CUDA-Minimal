#pragma once

#include <opencv2/core/mat.hpp>

// Structure definition for exchanging data between Host and Device
struct CudaImg
{
  uint3 m_size;             
  union {
      void   *m_p_void;     
      uchar1 *m_p_uchar1;  
      uchar3 *m_p_uchar3;   
      uchar4 *m_p_uchar4;   
  };
  void &cvtocudaRGB(cv::Mat &img)
  {
    m_size.x = img.size().width; 
    m_size.y = img.size().height;
    m_p_uchar3 = ( uchar3* ) img.data;
  };

  void &cvtocudaRGBA(cv::Mat &img)
  {
    m_size.x = img.size().width; 
    m_size.y = img.size().height; 
    m_p_uchar4 = ( uchar4* ) img.data;
  };

  __device__ uchar1 &getpixelBW(int y, int x)
  {
    return m_p_uchar1[y*m_size.x+x];
  }

  __device__ uchar3 &getpixelRGB(int y, int x)
  {
    return m_p_uchar3[y*m_size.x+x];
  }

  __device__ uchar4 &getpixelRGBA(int y, int x)
  {
    return m_p_uchar4[y*m_size.x+x];
  }

};