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
#include <vector>    //存取groundtruth信息
#include <fstream>

using namespace cv;
using namespace std;



cv::Rect split_line(string &line);   //分离字符串函数声明
//读取groundtruth信息
vector<cv::Rect> read_groundtruth(const string &groundtruth_txt,int &num_of_line)    //const常量才可以由字符串隐式转换
{
    vector<cv::Rect> groundtruth;            // vector<rect>  用来存groundtruth
    ifstream groundtruth_file;               // 文件对象
    groundtruth_file.open(groundtruth_txt);      //打开txt文件
    string line;         //当前行
    Rect rect_tmp;       //每一行搞成一个rect
    while(getline(groundtruth_file,line))
    {
        rect_tmp=split_line(line);    //分解字符串为RECT
        groundtruth.push_back(rect_tmp);    //压入vector
        num_of_line++;     
    }
    return groundtruth;
}

cv::Rect split_line(string &line)
{
    double pos[8];            //八个点
    int index=0;              //点的索引
    string tmp;               //暂存的string，来转换为double
    tmp.clear();              //清零
    for(auto l:line)          //遍历字符串，这里面是一个比较简单的字符串根据特定字符分离的一个算法
    {
        if(l==',')
        {
            pos[index]=stod(tmp);
            index++;
            tmp.clear();     //一定要记得清零
        }
        else
        {
            tmp+=l;
        }
    }
    pos[index]=stod(tmp);    //处理最后一个
    //四个点，对应矩形的四个点

    /* 我后来发现标注的点并不是遵循这样的规律，不一定一开始是左上角的点，这取决于当时标注的
    人先从哪个点开始点的，所以应该来使用坐标之间的大小关系来确定到底是哪个点
    cv::Point2f up_left(pos[0],pos[1]);
    cv::Point2f up_right(pos[2],pos[3]);
    cv::Point2f down_right(pos[4],pos[5]);
    cv::Point2f down_left(pos[6],pos[7]);
    //cout<<up_left<<" "<<up_right<<" "<<down_right<<" "<<down_left<<endl;


    int x=round(up_left.x);
    int y=round(up_left.y);
    int weidth=round(down_right.x-up_left.x);
    int height=round(down_right.y-up_left.y);
    cv::Rect res(x,y,weidth,height);
    */

   double xmin=min(pos[0],min(min(pos[2],pos[4]),pos[6]));
   double ymin=min(pos[1],min(min(pos[3],pos[5]),pos[7]));
   double xmax=max(pos[0],max(max(pos[2],pos[4]),pos[6]));
   double ymax=max(pos[1],max(max(pos[3],pos[5]),pos[7]));

   cv::Rect res(xmin,ymin,xmax-xmin,ymax-ymin);
   //cout<<res<<endl;
    
    return res;
}






extern "C" {




void kcf_test()
{
    printf("this is a kcf test code!!!\n");
    int num_of_line=0;  
    string path="VOT//car1//";
    
    //读取groundtruth信息
    vector<cv::Rect> groundtruth=read_groundtruth(path+"groundtruth.txt",num_of_line);

    vector<cv::Rect> track_res;
    track_res.push_back(groundtruth[0]);
   
    string zeros8="00000000";
    cv::Mat img=imread(path+"00000001.jpg");
    imshow("img",img);
    double all_time=0;
    KCFTracker tracker(true,true,false,false);    //构造
    tracker.init(groundtruth[0],img);      //初始化
    cv::rectangle(img,groundtruth[0],cv::Scalar(0,0,255));   //第一帧画框
    
   

    for(int i=2;i<num_of_line;i++)
    {
        string img_name=zeros8+std::to_string(i);
        string img_path=path+string(img_name.end()-8,img_name.end())+".jpg";
        
        cv::Mat frame=imread(img_path);    
        double start=static_cast<double>(getTickCount());
        cv::Rect res=tracker.update(frame);
        double time=((double)getTickCount()-start)/getTickFrequency();
        groundtruth.push_back(res);
        all_time+=time;
        //cout<<"fps\t"<<1./time<<endl;
        cout<<"ave_fps:\t"<<double(i-1)/all_time<<endl;
        cv::rectangle(frame,res,cv::Scalar(0,0,255));
        imshow("test",frame);
        waitKey(20);
    }
    

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
