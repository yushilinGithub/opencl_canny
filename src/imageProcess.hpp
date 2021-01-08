#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include<vector>
#include<string>
#ifdef APPLE
#include<OpenCL/cl.hpp>
#else
#include<CL/cl.hpp>
#endif
#include<iostream>

class imageProcessor{
    private:
        std::vector<cl::Platform> platform_;
        std::vector<cl::Device> device_;
        cl::Context context;
        cl::CommandQueue queue;

        cl::Buffer InputImage;
        cl::Buffer OutputImage;
        cl::Buffer thetaBuf;

        

        
        cl::Kernel guassian;
        cl::Kernel sobalkernel;
        cl::Kernel hysteresis;

        std::vector<size_t> maxGroupSize;

    public:
       
        cv::Mat img;
        cv::Mat outImg;
        cv::Mat theta;
        imageProcessor();
        
        void getImage(cv::Mat img);
        void getImageBuf();
        cl::Kernel loadKernel(std::string filename,std::string kernelname,size_t lsizeX=32,size_t lsizeY=32);
        void GuassianBlur();
        void sobal(float h_threshold,float l_threshold);
        void hyteresisfunc(float h_threshold,float l_threshold);
        void output();
        void finishJob();
};