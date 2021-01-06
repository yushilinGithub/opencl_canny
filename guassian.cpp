#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include"imageProcess.hpp"


int main(int argc, char **argv){
    int h_threshold = 70;
    int l_threshold = 20;
    std::string imagePath = "./image/theta1.jpg";
    imageProcessor imagePro = imageProcessor();
    cv::Mat image = cv::imread(imagePath,cv::IMREAD_COLOR);
    if(image.empty()){
        std::cout<<"image Path : "<<imagePath<<"did not exist"<<std::endl; 
        exit(1);       
    }

    imagePro.getImage(image);
    imagePro.getImageBuf();
    imagePro.GuassianBlur();
    imagePro.sobal(h_threshold,l_threshold);
    imagePro.hyteresisfunc(h_threshold,l_threshold);
    imagePro.output();
    imagePro.finishJob();
    
    if(!imagePro.img.empty()){
        cv::imshow("theta.jpg",imagePro.img);

        cv::waitKey(0);
    }
}