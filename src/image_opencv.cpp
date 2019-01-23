#ifdef OPENCV

//下面这三个都是为了KCF来的
#include "ffttools.hpp"    //这个是为了fhog来做傅里叶变换的
#include "recttools.hpp"
#include "kcftracker.hpp"
#include <iostream>
#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image.h"
#include <string> 

using namespace cv;
using namespace std;







extern "C" {

void kcf_test()
{
    string zeros8="00000000";
    
    
    printf("this is a kcf test code!!!\n");
    cv::Mat img=imread("VOT//00000001.jpg");

    KCFTracker tracker(true,true,false,false);    //构造
    tracker.init(cv::Rect(444,332,10,10),img);  //初始化
    cv::rectangle(img,cv::Rect(444,332,48,40),cv::Scalar(0,0,255));
    imshow("lll",img);
    double all_time=0;

    for(int i=2;i<250;i++)
    {
        string img_name=zeros8+std::to_string(i);
        string img_path="VOT//"+string(img_name.end()-8,img_name.end())+".jpg";
        
        cv::Mat frame=imread(img_path);  
        double start=static_cast<double>(getTickCount());
        cv::Rect res=tracker.update(frame);
        double time=((double)getTickCount()-start)/getTickFrequency();
        all_time+=time;
        //cout<<"fps\t"<<1./time<<endl;
        //cout<<"ave_fps:\t"<<double(i-1)/all_time<<endl;
        cv::rectangle(frame,res,cv::Scalar(0,0,255));
        imshow("test",frame);
        waitKey(20);
        

    }

    
    
   
    //imshow("guas",tracker._prob);
    waitKey(0);

}

IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

//IPL是opencv的C接口，这里就是一个图像的 转换
image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}

//image GO Mat，Mat是OPENCV的新接口
Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

image mat_to_image(Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}

void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return (void *) cap;
}

image get_image_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_image(0,0,0);
    return mat_to_image(m);
}

image load_image_cv(char *filename, int channels)
{
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m;
    m = imread(filename, flag);
    if(!m.data){
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    image im = mat_to_image(m);
    return im;
}

int show_image_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);    //image转换成MAT
    imshow(name, m);         //显示图片，
    int c = waitKey(ms);     //演示
    if (c != -1) c = c%256;     //对256取余，这个是返回waiktey的返回值，ascii码应该是
    return c;
}

void make_window(char *name, int w, int h, int fullscreen)
{
    namedWindow(name, WINDOW_NORMAL);    //这个是一个可以调整大小的windows,用鼠标来调整
    if (fullscreen) {
        setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}

}

#endif
