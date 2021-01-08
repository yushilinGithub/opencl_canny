#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include"imageProcess.hpp"


int main(int argc, char **argv){
    float h_threshold = 40;
    float l_threshold = 20;
    std::string imagePath = "./image/theta1.jpg";
    imageProcessor imagePro = imageProcessor();
    cv::Mat image = cv::imread(imagePath,cv::IMREAD_COLOR);
    cv::Mat gray;
    cv::Mat canny;
    if(image.empty()){
        std::cout<<"image Path : "<<imagePath<<"did not exist"<<std::endl; 
        exit(1);       
    }
    cv::cvtColor(image,gray,cv::COLOR_RGB2GRAY);


    imagePro.getImage(image);
    imagePro.getImageBuf();
    imagePro.GuassianBlur();
    imagePro.sobal(h_threshold,l_threshold);
    imagePro.hyteresisfunc(h_threshold,l_threshold);
    imagePro.output();
    imagePro.finishJob();

    cv::imshow("theta.jpg",imagePro.img);
    cv::imshow("image",image);

    cv::waitKey(0);

}