#define __CL_ENABLE_EXCEPTIONS

#include<iostream>
#include<fstream>
#include<boost/format.hpp>

#include "imageProcess.hpp"
imageProcessor::imageProcessor(){
    cl_int error;

    error = cl::Platform::get(&platform_);
    if(error!=CL_SUCCESS){
        std::cout<<"Couldn't get platform"<<std::endl;
    }
    error = platform_[0].getDevices(CL_DEVICE_TYPE_GPU,&device_);
    maxGroupSize.assign(device_.size(),0);
    for(int i;i<device_.size();i++){
        error = device_[i].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE,&maxGroupSize[i]);
        if(error!=CL_SUCCESS){
            std::cout<<"get max group size info error"<<std::endl;
            exit(1);     
        }
        std::cout<<"maxGroupSize:"<<maxGroupSize[i]<<std::endl;
    }
    if(error!=CL_SUCCESS){
        std::cout<<"Couldn't get device"<<std::endl;
    }
    context = cl::Context(device_);
    queue = cl::CommandQueue(context,device_[0]);

    guassian = loadKernel("./kernel/guassian.cl","guassian");   
    sobalkernel = loadKernel("./kernel/sobal.cl","stage1_with_sobel");
    hysteresis = loadKernel("./kernel/hysteresis.cl","stage2_hysteresis",32,32/8);
}
cl::Kernel imageProcessor::loadKernel(std::string filename,std::string kernelname,size_t lsizeX,size_t lsizeY){
    std::ifstream c_file(filename);
    if(!c_file.good()){
        std::cout<<"couldn't open the file:"<<filename<<std::endl;
        exit(1);
    }
    std::string c_string(std::istreambuf_iterator<char>(c_file),(std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1,std::make_pair(c_string.c_str(),c_string.length()+1));
    cl::Program program(context,source);

    try{
        boost::format fmt("-D GRP_SIZEX=%1% -D GRP_SIZEY=%2%");
        fmt%lsizeX%lsizeY;
        program.build(device_,fmt.str().c_str());
    }catch(cl::Error e){
        std::cout<<"build status:\t:";
        std::cout<<program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device_[0])<<std::endl;
        std::cout<<"build log:\t";
        std::cout<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_[0])<<std::endl;
        std::cout<<"build option\t";
        std::cout<<program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device_[0])<<std::endl;
    }
    return cl::Kernel(program,kernelname.c_str());
}
void imageProcessor::getImage(cv::Mat colorimg){
    int l_size = 32;
    if(img.elemSize()!=1){
        cv::cvtColor(colorimg,img,cv::COLOR_BGR2GRAY);
    }else if(img.channels()==1){
        colorimg.copyTo(img);
    }
    outImg  = cv::Mat(img.rows,img.cols,CV_8UC1);
    theta = cv::Mat(img.rows,img.cols,CV_8UC1);
}

void imageProcessor::getImageBuf(){
    size_t size = img.rows*img.cols;

    try{
        InputImage = cl::Buffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,size,img.data);
        OutputImage = cl::Buffer(context,CL_MEM_COPY_HOST_PTR|CL_MEM_READ_WRITE,size,outImg.data);

    }catch(cl::Error e){
        std::cerr<<"error:"<<e.what()<<"\t"<<e.err()<<std::endl;
    }
}

void imageProcessor::GuassianBlur(){
    cl_int error;
        size_t lsizeX=32;
        size_t lsizeY = maxGroupSize[0]/32;

        if(lsizeY==0){
            lsizeX = 16;
            lsizeY = maxGroupSize[0]/16;
        }
        if(lsizeY==0){
            lsizeY = 1;
        }
        

        error = guassian.setArg(0,InputImage);
        if(error!=CL_SUCCESS){
            std::cerr<<"guassian: Couldn't set first argument"<<std::endl;
            
        }
        error = guassian.setArg(1,OutputImage);
        if(error!=CL_SUCCESS){
            std::cerr<<"guassian: Couldn't set second argument"<<std::endl;
        }

        guassian.setArg(2,img.rows);
        guassian.setArg(3,img.cols);
        
        size_t global_sizeY,global_sizeX;
        global_sizeY = (img.rows/lsizeY)*lsizeY;
        global_sizeX = (img.cols/lsizeX)*lsizeX;

        if(global_sizeY<img.rows) global_sizeY+=lsizeY;
        if(global_sizeX<img.cols) global_sizeX+=lsizeX;

        error = queue.enqueueNDRangeKernel(guassian,cl::NDRange(0,0),cl::NDRange(global_sizeY,global_sizeX),cl::NDRange(lsizeY,lsizeX));
        if(error!=CL_SUCCESS){
            std::cout<<"guassian couldn't enqueue kernel"<<std::endl;
            exit(1);
        }
}

void imageProcessor::sobal(float h_threshold,float l_threshold){
    
    cl_int error;
    size_t lsizeX=32;
    size_t lsizeY = maxGroupSize[0]/32;



    if(lsizeY==0){
        lsizeX = 16;
        lsizeY = maxGroupSize[0]/16;
    }
    if(lsizeY==0){
        lsizeY = 1;
    }

    if(h_threshold>0 && l_threshold>0){
        h_threshold = h_threshold*h_threshold;
        l_threshold = l_threshold*l_threshold;
    }
        
    error = sobalkernel.setArg(0,OutputImage);
    if(error!=CL_SUCCESS){
        std::cout<<"sobal set first argument fault"<<std::endl;
    }
    error = sobalkernel.setArg(1,InputImage);
    if(error!=CL_SUCCESS){
        std::cout<<"sobal set second argument fault"<<std::endl;
    }

    error = sobalkernel.setArg(2,img.rows);
    error = sobalkernel.setArg(3,img.cols);
    error = sobalkernel.setArg(4,h_threshold);
    error = sobalkernel.setArg(5,l_threshold);
    
    size_t global_sizeY,global_sizeX;
    global_sizeY = (img.rows/lsizeY)*lsizeY;
    global_sizeX = (img.cols/lsizeX)*lsizeX;

    if(global_sizeY<img.rows) global_sizeY+=lsizeY;
    if(global_sizeX<img.cols) global_sizeX+=lsizeX;

    error = queue.enqueueNDRangeKernel(sobalkernel,cl::NDRange(0,0),cl::NDRange(global_sizeY,global_sizeX),cl::NDRange(lsizeY,lsizeX));
    if(error!=CL_SUCCESS){
        std::cout<<"guassian couldn't enqueue kernel"<<std::endl;
        exit(1);
    }
}

void imageProcessor::hyteresisfunc(float h_threshold,float l_threshold){
        size_t lsizeX=32;
        size_t lsizeY = maxGroupSize[0]/32;

        if(lsizeY==0){
            lsizeX = 16;
            lsizeY = maxGroupSize[0]/16;
        }
        if(lsizeY==0){
            lsizeY = 1;
        }
    
    lsizeY = lsizeY/8;
    size_t global_sizeY = (img.rows+7)/8,global_sizeX;

    global_sizeY = (global_sizeY/lsizeY)*lsizeY;
    global_sizeX = (img.cols/lsizeX)*lsizeX;

    if(global_sizeY<img.rows) global_sizeY+=lsizeY;
    if(global_sizeX<img.cols) global_sizeX+=lsizeX;
        
    cl_int error;


    try{
        error = hysteresis.setArg(0,InputImage);
        error = hysteresis.setArg(1,img.rows);
        error = hysteresis.setArg(2,img.cols);
        
        error = queue.enqueueNDRangeKernel(hysteresis,cl::NDRange(0,0),\
                                            cl::NDRange(global_sizeY,global_sizeX),\
                                            cl::NDRange(lsizeY,lsizeX));
        if(error!=CL_SUCCESS){
            std::cout<<"hysteresis couldn't enqueue kernel" <<std::endl;
            exit(1);
        }
        }catch(cl::Error e){
            std::cerr<<" hysteresis error : "<<e.what()<<" : "<<e.err()<<std::endl;
            exit(1);
        }
}
void imageProcessor::output(){
    cl_int error;
    try{
        error = queue.enqueueReadBuffer(InputImage,CL_TRUE,0,size_t(outImg.rows*outImg.cols),img.data);
    }catch(cl::Error e){
        std::cerr<<"enqueue read buffer error"<<e.what()<<":"<<e.err()<<std::endl;
    }
}
void imageProcessor::finishJob(){
    try{
        queue.finish();
    }catch(cl::Error e){
        std::cerr<<"finish error"<<e.what()<<":"<<e.err()<<std::endl;
    }
}