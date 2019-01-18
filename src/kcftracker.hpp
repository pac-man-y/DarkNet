
#pragma once


const int nClusters = 15;
float dataa[nClusters][3] = {
{161.317504, 127.223401, 128.609333},
{142.922425, 128.666965, 127.532319},
{67.879757, 127.721830, 135.903311},
{92.705062, 129.965717, 137.399500},
{120.172257, 128.279647, 127.036493},
{195.470568, 127.857070, 129.345415},
{41.257102, 130.059468, 132.675336},
{12.014861, 129.480555, 127.064714},
{226.567086, 127.567831, 136.345727},
{154.664210, 131.676606, 156.481669},
{121.180447, 137.020793, 153.433743},
{87.042204, 137.211742, 98.614874},
{113.809537, 106.577104, 157.818094},
{81.083293, 170.051905, 148.904079},
{45.015485, 138.543124, 102.402528}};


#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"

#include <opencv2/opencv.hpp>
#include <string>



//Tracker的定义，这里相当于就只是做了一个接口！一个init一个update
class Tracker
{
public:
    Tracker()  {}
   virtual  ~Tracker() { }

    virtual void init(const cv::Rect &roi, cv::Mat image) = 0;
    virtual cv::Rect  update( cv::Mat image)=0;


protected:
    cv::Rect_<float> _roi;
};





class KCFTracker : public Tracker
{
public:
    // Constructor  构造函数
    KCFTracker(bool hog = true, bool fixed_window = true, bool multiscale = true, bool lab = true);

    // Initialize tracker   初始化跟踪器
    virtual void init(const cv::Rect &roi, cv::Mat image);
    
    // Update position based on the new frame   更新跟踪器，只需要当前帧图像就可以，返回Rect
    virtual cv::Rect update(cv::Mat image);

    float interp_factor; // 线性插值系数linear interpolation factor for adaptation
    float sigma; // 高斯核系数gaussian kernel bandwidth
    float lambda; // 正则化系数regularization
    int cell_size; // HOG cell size
    int cell_sizeQ; // cell size^2, to avoid repeated operations
    float padding; // padding sz 一般1-2.5extra area surrounding the target
    float output_sigma_factor; //高斯标签系数bandwidth of gaussian target
    int template_size; // 当前size（或者是模板sz） template size
    float scale_step; // 尺度步长 scale step for multi-scale estimation
    float scale_weight;  // 尺度权重，加强当前选择的权重to downweight detection scores of other scales for added stability

//protected:
    // 检测函数 Detect object in the current frame.
    cv::Point2f detect(cv::Mat z, cv::Mat x, float &peak_value);

    // 训练函数 train tracker with a single image
    void train(cv::Mat x, float train_interp_factor);

    // 高斯相关核函数（核技巧提供可分性） Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
    cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2);

    // 创建高斯标签（二维单峰高斯）Create Gaussian Peak. Function called only in the first frame.
    cv::Mat createGaussianPeak(int sizey, int sizex);

    // 创建子窗口（并获得features？）Obtain sub-window from image, with replication-padding and extract features
    cv::Mat getFeatures(const cv::Mat & image, bool inithann, float scale_adjust = 1.0f);

    // 创建汉明窗，这个是为了减少傅里叶变换的边界效应Initialize Hanning window. Function called only in the first frame.
    void createHanningMats();

    // 这个是获取二维此窗口的一个peak（峰值，也就是说检测的位置应该是）Calculate sub-pixel peak for one dimension
    float subPixelPeak(float left, float center, float right);

    cv::Mat _alphaf;     //就是原式中的alphaf  但是是在频域总的
    cv::Mat _prob;       //高斯峰的频域，这个使用createGaussianPeak()来创建的时候最后返回的是做过傅里叶变换的
    cv::Mat _tmpl;   
    cv::Mat _num;
    cv::Mat _den;
    cv::Mat _labCentroids;

private:
    int size_patch[3];
    cv::Mat hann;
    cv::Size _tmpl_sz;
    float _scale;
    int _gaussian_size;
    bool _hogfeatures;
    bool _labfeatures;
};





//跟踪器的构造函数，这个里面实际上是没有做什么实质性的操作和运算，只是初始化了一些参数和flag
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{

    // Parameters equal in all cases
    lambda = 0.0001;     
    padding = 2.5; 
    //output_sigma_factor = 0.1;
    output_sigma_factor = 0.125;


    if (hog) {    // HOG   如果用hog的话
        // VOT
        interp_factor = 0.012;  
        sigma = 0.6; 
        // TPAMI
        //interp_factor = 0.02;
        //sigma = 0.5; 
        cell_size = 4;     //hog的cell_size
        _hogfeatures = true;    //标志位

        //lab这个是什么意思？，我确实没有看懂
        if (lab) {
            interp_factor = 0.005;
            sigma = 0.4; 
            //output_sigma_factor = 0.025;
            output_sigma_factor = 0.1;

            _labfeatures = true;
            _labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &dataa);
            cell_sizeQ = cell_size*cell_size;
        }
        
            _labfeatures = false;
        
    }
    else {   // RAW
        interp_factor = 0.075;
        sigma = 0.2; 
        cell_size = 1;
        _hogfeatures = false;
        /*
        if (lab) {
            printf("Lab features are only used with HOG features.\n");
            _labfeatures = false;
        }
        */
    }


    if (multiscale) { // multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1.05;
        scale_weight = 0.95;
        if (!fixed_window) {
            //printf("Multiscale does not support non-fixed window.\n");
            fixed_window = true;
        }
    }
    else if (fixed_window) {  // fit correction without multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1;
    }
    else {
        template_size = 1;
        scale_step = 1;
    }
}



//初始化跟踪器，使用ROI和IMAGE  Initialize tracker 
void KCFTracker::init(const cv::Rect &roi, cv::Mat image)
{
    _roi = roi;       //roi设置
    assert(roi.width >= 0 && roi.height >= 0);    //长和宽必须都大于0，否则终止程序并报错，主要是调试的时候用
    _tmpl = getFeatures(image, 1);         //获取特征图
    _prob = createGaussianPeak(size_patch[0], size_patch[1]);    //创建高斯峰，这个只在第一帧的时候创建
    _alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));   //创建alphaf，是32位复数二通道，用0初始化
    //_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    train(_tmpl, 1.0); // train with initial frame     用第一帧来训练
 }
// Update position based on the new frame
cv::Rect KCFTracker::update(cv::Mat image)
{
    //四个边界条件处理一下，不能越过边界
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;


    float peak_value;   //峰值，当一个返回值去传入，真正的返回值是res(POINT2F)
    cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value);

    if (scale_step != 1) {
        // Test at a smaller _scale
        float new_peak_value;
        cv::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);

        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale /= scale_step;
            _roi.width /= scale_step;
            _roi.height /= scale_step;
        }

        // Test at a bigger _scale
        new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);

        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale *= scale_step;
            _roi.width *= scale_step;
            _roi.height *= scale_step;
        }
    }

    // Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale);
    _roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

    assert(_roi.width >= 0 && _roi.height >= 0);
    cv::Mat x = getFeatures(image, 0);
    train(x, interp_factor);

    return _roi;
}


// Detect object in the current frame.
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value)
{
    using namespace FFTTools;    //detect其实也主要做的是这个工作

    cv::Mat k = gaussianCorrelation(x, z);     //使用核技巧
    cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));

    //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
    peak_value = (float) pv;

    //subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < res.cols-1) {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x-1), peak_value, res.at<float>(pi.y, pi.x+1));
    }

    if (pi.y > 0 && pi.y < res.rows-1) {
        p.y += subPixelPeak(res.at<float>(pi.y-1, pi.x), peak_value, res.at<float>(pi.y+1, pi.x));
    }

    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;

    return p;
}

// train tracker with a single image
void KCFTracker::train(cv::Mat x, float train_interp_factor)
{
    using namespace FFTTools;

    cv::Mat k = gaussianCorrelation(x, x);
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));
    
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;


    /*cv::Mat kf = fftd(gaussianCorrelation(x, x));
    cv::Mat num = complexMultiplication(kf, _prob);
    cv::Mat den = complexMultiplication(kf, kf + lambda);
    
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _num = (1 - train_interp_factor) * _num + (train_interp_factor) * num;
    _den = (1 - train_interp_factor) * _den + (train_interp_factor) * den;

    _alphaf = complexDivision(_num, _den);*/

}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must also be periodic (ie., pre-processed with a cosine window).
// 使用核技巧做核相关矩阵，特征图都应该是M*N的，但是FHOG中getfeature返回的是一行，所以这里需要做重新reshape
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    //做和相关矩阵
    using namespace FFTTools;
    cv::Mat c = cv::Mat( cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0) );
    // HOG features
    if (_hogfeatures) {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < size_patch[2]; i++)   //size_patch[2]里面存的是numOfFeature
        {
            x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
            x1aux = x1aux.reshape(1, size_patch[0]);   //第i行拿出来
            x2aux = x2.row(i).reshape(1, size_patch[0]);  //把x2的第i行也拿出来
            
            std::cout<<"xlaux_sz:\t"<<x1aux.size()<<std::endl;
            
            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true); 
            caux = fftd(caux, true);
            rearrange(caux);
            caux.convertTo(caux,CV_32F);
            c = c + real(caux);
        }
    }
    // Gray features
    else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }
    cv::Mat d; 
    cv::max(( (cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0])- 2. * c) / (size_patch[0]*size_patch[1]*size_patch[2]) , 0, d);

    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k);
    return k;
}

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

    float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
        }
    return FFTTools::fftd(res);
}

// Obtain sub-window from image, with replication-padding and extract features
// 获取子窗口，并返回特征图，31*(sizex*sizey)维度
cv::Mat KCFTracker::getFeatures(const cv::Mat & image, bool inithann, float scale_adjust)
{
    cv::Rect extracted_roi;   //padding之后的ROI

    //这就是中心点了，然后根据padding来获取subwindow
    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;

    // 如果要hann窗的话
    if (inithann) {
        //
        int padded_w = _roi.width * padding;
        int padded_h = _roi.height * padding;     
        
        //尺寸的一些变化
        if (template_size > 1) {  // Fit largest dimension to the given template size
            if (padded_w >= padded_h)  //fit to width
                _scale = padded_w / (float) template_size;
            else
                _scale = padded_h / (float) template_size;

            _tmpl_sz.width = padded_w / _scale;
            _tmpl_sz.height = padded_h / _scale;
        }
        else {  //No template size given, use ROI size，没有给模板的话就用的是template
            _tmpl_sz.width = padded_w;
            _tmpl_sz.height = padded_h;
            _scale = 1;
            // original code from paper:
            /*if (sqrt(padded_w * padded_h) >= 100) {   //Normal size
                _tmpl_sz.width = padded_w;
                _tmpl_sz.height = padded_h;
                _scale = 1;
            }
            else {   //ROI is too big, track at half size
                _tmpl_sz.width = padded_w / 2;
                _tmpl_sz.height = padded_h / 2;
                _scale = 2;
            }*/
        }
        
        // hog特征
        if (_hogfeatures) {
            // Round to cell size and also make it even，变成偶数，这个肯定是引入新的数据了，先除以2*cell_size，然后在乘，最后加上cell_size*2
            _tmpl_sz.width = ( ( (int)(_tmpl_sz.width / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
            _tmpl_sz.height = ( ( (int)(_tmpl_sz.height / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
        }
        //如果不用hog特征的话就直接做了，还是把尺寸变成偶数，使得一些折半计算的操作更简单一些，比如获取一般尺寸的时候
        else {  //Make number of pixels even (helps with some logic involving half-dimensions)
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
            _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
        }
    }

    //extracted_roi区域的大小，
    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;

    // center roi with new size  新SIZE的中心点
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    cv::Mat FeaturesMap;     //特征图
    //这里是获取根据extracted_roi和image来从原图中截取subwindow，这个函数在RECTtools头文件里
    cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
    
    //如果不等于的话是需要resize的，采取什么样的插值方式就默认的了，这样其实取了很少一部分东西。
    if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
        cv::resize(z, z, _tmpl_sz);
    }   
    

    // HOG features  获取FHOG特征，这个就要FHOG.H里的东西了，这部分应该是最难的，但是可以不看哈哈……
    if (_hogfeatures) {
        IplImage z_ipl = z;    //Mat可以直接赋值给IPLIMAGE，看来是可以的
        CvLSVMFeatureMapCaskade *map;     //包括size，num和一个float的数组，里面存的就是hog的特征
        getFeatureMaps(&z_ipl, cell_size, &map);
        normalizeAndTruncate(map,0.2f);    //归一化和截断，这个在Fhog里的论文里也有写，可以参见这里：https://www.jianshu.com/p/69a3e39c51f9
        PCAFeatureMaps(map);        //PCA降维，本来（9+18)*4个主成分，按照PCA降维之后，会得到27+4一共31维的特征
        size_patch[0] = map->sizeY;
        size_patch[1] = map->sizeX;
        size_patch[2] = map->numFeatures;
        //这里的numFeatures就应该是31了

        //这里是把每一维的特征当做一行来存储到Mat里，一共是31行，每行有map->sizeX*map->sizeY这么多个数，32维浮点数
        FeaturesMap = cv::Mat(cv::Size(map->numFeatures,map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
        FeaturesMap = FeaturesMap.t();    //转置，恩就是这样的
        freeFeatureMapObject(&map);   //map是C代码，所以这部分的内存需要手动释放

        // Lab features 是否需要加上lab特征，这里的意思应该是lab颜色特征
        if (_labfeatures) {
            cv::Mat imgLab;
            cvtColor(z, imgLab, CV_BGR2Lab);
            unsigned char *input = (unsigned char*)(imgLab.data);

            // Sparse output vector
            cv::Mat outputLab = cv::Mat(_labCentroids.rows, size_patch[0]*size_patch[1], CV_32F, float(0));

            int cntCell = 0;
            // Iterate through each cell
            for (int cY = cell_size; cY < z.rows-cell_size; cY+=cell_size){
                for (int cX = cell_size; cX < z.cols-cell_size; cX+=cell_size){
                    // Iterate through each pixel of cell (cX,cY)
                    for(int y = cY; y < cY+cell_size; ++y){
                        for(int x = cX; x < cX+cell_size; ++x){
                            // Lab components for each pixel
                            float l = (float)input[(z.cols * y + x) * 3];
                            float a = (float)input[(z.cols * y + x) * 3 + 1];
                            float b = (float)input[(z.cols * y + x) * 3 + 2];

                            // Iterate trough each centroid
                            float minDist = FLT_MAX;
                            int minIdx = 0;
                            float *inputCentroid = (float*)(_labCentroids.data);
                            for(int k = 0; k < _labCentroids.rows; ++k){
                                float dist = ( (l - inputCentroid[3*k]) * (l - inputCentroid[3*k]) )
                                           + ( (a - inputCentroid[3*k+1]) * (a - inputCentroid[3*k+1]) ) 
                                           + ( (b - inputCentroid[3*k+2]) * (b - inputCentroid[3*k+2]) );
                                if(dist < minDist){
                                    minDist = dist;
                                    minIdx = k;
                                }
                            }
                            // Store result at output
                            outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ; 
                            //((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ; 
                        }
                    }
                    cntCell++;
                }
            }
            // Update size_patch[2] and add features to FeaturesMap
            size_patch[2] += _labCentroids.rows;
            FeaturesMap.push_back(outputLab);
        }
    }
    else {
        FeaturesMap = RectTools::getGrayImage(z);
        FeaturesMap -= (float) 0.5; // In Paper;
        size_patch[0] = z.rows;
        size_patch[1] = z.cols;
        size_patch[2] = 1;  
    }
    
    if (inithann) {
        createHanningMats();
    }
    FeaturesMap = hann.mul(FeaturesMap);     //加上汉明窗
    return FeaturesMap;     //返回
}
    
// Initialize Hanning window. Function called only in the first frame.汉明窗的创建只在第一帧
void KCFTracker::createHanningMats()
{   
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1],1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,size_patch[0]), CV_32F, cv::Scalar(0)); 

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
    // HOG features
    if (_hogfeatures) {
        cv::Mat hann1d = hann2d.reshape(1,1); // Procedure do deal with cv::Mat multichannel bug
        
        hann = cv::Mat(cv::Size(size_patch[0]*size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
        for (int i = 0; i < size_patch[2]; i++) {
            for (int j = 0; j<size_patch[0]*size_patch[1]; j++) {
                hann.at<float>(i,j) = hann1d.at<float>(0,j);
            }
        }
    }
    // Gray features
    else {
        hann = hann2d;
    }
}

// Calculate sub-pixel peak for one dimension
float KCFTracker::subPixelPeak(float left, float center, float right)
{   
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;
    
    return 0.5 * (right - left) / divisor;
}