#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv )
{
    VideoCapture cap("/dev/video0", cv::CAP_V4L2);
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    cout << "ok\n";
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('G','R','E','Y'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 400);
    cap.set(cv::CAP_PROP_CONVERT_RGB, 0);
    //cap.set(cv::CAP_PROP_FPS, 5);
    //cout << "fps:" << cap.get(cv::CAP_PROP_FPS) << "\n";
    //cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
    //cap.set(cv::CAP_PROP_EXPOSURE, 800);
                
    //system("v4l2-ctl -c exposure=800,frame_rate=5");                

    Mat cam0 = (Mat_<double>(3,3) <<
        4.5871894208851103e+02, 0., 3.3657516241366756e+02, 0.,
        4.5873145101846637e+02, 2.1628074178739530e+02, 0., 0., 1); 
    Mat dist0 = (Mat_<double>(5,1) <<
        5.7076163517889016e-02, -4.7449852647681277e-02,
       -3.3333133411083044e-03, 3.5791943710588030e-04, 0.);
    Mat cam1 = (Mat_<double>(3,3) << 
        4.5812704270150437e+02, 0., 3.2301894953617500e+02, 0.,
        4.5787816672137893e+02, 2.2334505470968028e+02, 0., 0., 1.);
    Mat dist1 = (Mat_<double>(5,1) << 
        5.3889678894950083e-02, -5.3416446280919003e-02,
       -4.7029682310199220e-03, 1.0284166453383803e-03, 0.);
    Mat R1 =  (Mat_<double>(3,3) << 
        9.9994868632879330e-01, -7.3898639851504491e-03,
       -6.9292582288108919e-03, 7.4219106040843393e-03,
       9.9996182819660484e-01, 4.6105740085947630e-03,
       6.8949222117111221e-03, -4.6617657582428652e-03,
       9.9996536339400766e-01);
    Mat R2 = (Mat_<double>(3,3) << 
       9.9986633891259658e-01, -9.8605780086776576e-03,
       1.3041215850347026e-02, 9.9209361145806835e-03,
       9.9994033556126416e-01, -4.5716894291580878e-03,
       -1.2995358253275224e-02, 4.7004594414854616e-03,
       9.9990450861315161e-01);
    Mat P1 = (Mat_<double>(3,4) <<
        4.9192885640324891e+02, 0., 3.2796970748901367e+02, 0., 0.,
       4.9192885640324891e+02, 2.1781100463867188e+02, 0., 0., 0., 1.,
       0.);
    Mat P2 = (Mat_<double>(3,4) <<
        4.9192885640324891e+02, 0., 3.2796970748901367e+02,
       -4.9456374389412197e+01, 0., 4.9192885640324891e+02,
       2.1781100463867188e+02, 0., 0., 0., 1., 0.);
    Mat Q = (Mat_<float>(4,4) <<
        1., 0., 0., -3.2796970748901367e+02, 0., 1., 0.,
       -2.1781100463867188e+02, 0., 0., 0., 4.9192885640324891e+02, 0.,
       0., 9.9467229952983143e+00, 0.);
    cv::Mat cam0_map1, cam0_map2, cam1_map1, cam1_map2;
    initUndistortRectifyMap(cam0, dist0, R1, P1, cv::Size2i(640,400), CV_32FC1, cam0_map1, cam0_map2);
    initUndistortRectifyMap(cam1, dist1, R2, P2, cv::Size2i(640,400), CV_32FC1, cam1_map1, cam1_map2);
    cv::cuda::GpuMat g_cam0_map1(cam0_map1);
    cv::cuda::GpuMat g_cam0_map2(cam0_map2);
    cv::cuda::GpuMat g_cam1_map1(cam1_map1);
    cv::cuda::GpuMat g_cam1_map2(cam1_map2);

    void *unified_ptr;
    cudaMallocManaged(&unified_ptr, 1280*400);
    cv::Mat frame(400, 1280, CV_8UC1, unified_ptr);
    cv::cuda::GpuMat g_frame(400, 1280, CV_8UC1, unified_ptr);

    void *unified_ptr_l, *unified_ptr_r;    
    cudaMallocManaged(&unified_ptr_l, 640*400);
    cv::cuda::GpuMat g_frame_l_rect(400,640, CV_8UC1, unified_ptr_l);
    cv::Mat left(400, 640, CV_8UC1, unified_ptr_l);
    cudaMallocManaged(&unified_ptr_r, 640*400);
    cv::cuda::GpuMat g_frame_r_rect(400,640, CV_8UC1, unified_ptr_r);
    cv::Mat right(400, 640, CV_8UC1, unified_ptr_r);

    /*void *unified_ptr_disp;
    cudaMallocManaged(&unified_ptr_disp, 640*400*2);
    cv::cuda::GpuMat g_disp_map(400, 640, CV_16S, unified_ptr_disp);
    cv::Mat disp_map(400, 640, CV_16S, unified_ptr_disp);*/
    cv::cuda::GpuMat g_disp_map(400, 640, CV_16SC1);
    cv::cuda::GpuMat g_disp_map2(400, 640, CV_32FC1);
    cv::cuda::GpuMat g_3d_img(400, 640, CV_32FC3);
    cv::cuda::GpuMat g_split_img[3];
    cv::cuda::GpuMat g_depth_img;
    cv::cuda::GpuMat g_depth_img_scaled;
    cv::Mat disp_map;

    cv::Ptr<cv::cuda::StereoSGM> sgbm = cv::cuda::createStereoSGM();

    //cv::cuda::Stream cuda_stream(cudaStreamNonBlocking);

    //cv::Ptr< cv::cuda::Filter > filter = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 1, 1, 1, 1, cv::BORDER_DEFAULT);

    cv::cuda::GpuMat g_frame_l, g_frame_r;
    cv::namedWindow("example", cv::WINDOW_AUTOSIZE);
    int n=0;
    char f_name[32];
    bool not_init=true;
    while(1) {
        if (cap.grab()) {
            cap.retrieve(frame);
            if (frame.empty()) {
                cout << "blank frame?!\n";
                break;
            }
            g_frame_l = g_frame.colRange(g_frame.cols / 2, g_frame.cols);
            g_frame_r = g_frame.colRange(0, g_frame.cols / 2);
            cv::cuda::remap(g_frame_l, g_frame_l_rect, g_cam0_map1, g_cam0_map2, INTER_LINEAR);
            cv::cuda::remap(g_frame_r, g_frame_r_rect, g_cam1_map1, g_cam1_map2, INTER_LINEAR);
            //auto t_a = chrono::high_resolution_clock::now();
            sgbm->compute(g_frame_l_rect, g_frame_r_rect, g_disp_map);
            cv::cuda::multiply(g_disp_map, cv::Scalar(0.0625), g_disp_map2, 1,CV_32FC1);
            cv::cuda::reprojectImageTo3D(g_disp_map2, g_3d_img, Q, 3);
            //auto t_b = chrono::high_resolution_clock::now();
            //cout<<"with cuda:"<< chrono::duration_cast<chrono::microseconds>(t_b-t_a).count()<<"us"<<endl;
            //cuda_stream.waitForCompletion();
            cv::cuda::split(g_3d_img, g_split_img);
            //double min, max;
            //cv::cuda::minMax(g_split_img[2], &min, &max);
            //cout << min << "," << max << endl;
            cv::cuda::threshold(g_split_img[2], g_depth_img, 0, 0, THRESH_TOZERO);
            cv::cuda::threshold(g_depth_img, g_depth_img, 5 , 0, THRESH_TRUNC);
            g_depth_img.convertTo(g_depth_img_scaled, CV_8UC1, 50);
            
            //cv::cuda::normalize(g_depth_img, g_depth_img_norm, 0, 255, NORM_MINMAX, CV_8UC1);
            //filteri->apply(g_frame_l, g_frame_l);
            //right = frame.colRange(0, frame.cols / 2);
            //left = frame.colRange(frame.cols / 2, frame.cols);
            g_depth_img_scaled.download(disp_map);
            imshow("example", disp_map);
            if (not_init) {
                system("v4l2-ctl -c exposure=900,frame_rate=30");
                //cap.set(cv::CAP_PROP_FPS, 5);
                //cap.set(cv::CAP_PROP_EXPOSURE, 800);
                not_init=false;
            }
        } else cout << "no frame\n";
        int c = waitKey(1);
        if (c == 'q') break;
        else if (c == 'f') {
            imwrite("left"+std::to_string(n)+".png", left);
            imwrite("right"+std::to_string(n)+".png", right);
            n++;
        }
    }

    cap.release();
    cudaFree(&unified_ptr);
    cudaFree(&unified_ptr_l);
    cudaFree(&unified_ptr_r);
    //cudaFree(&unified_ptr_disp);
    destroyAllWindows();

    return 0;
}
