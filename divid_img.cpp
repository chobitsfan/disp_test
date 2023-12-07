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

    cv::Mat cam0 = (cv::Mat_<double>(3,3) <<
        4.5871894208851108e+02, 0., 3.3657516241366756e+02, 0.,
        4.5873145101846643e+02, 2.1628074178739521e+02, 0., 0., 1.);
    cv::Mat dist0 = (cv::Mat_<double>(5,1) <<
        5.7076163517889071e-02, -4.7449852647679529e-02,
       -3.3333133411082233e-03, 3.5791943710580180e-04, 0.);
    cv::Mat cam1 = (cv::Mat_<double>(3,3) <<
        4.5835888364887256e+02, 0., 3.2070900760895131e+02, 0.,
        4.5836189927937050e+02, 2.2369740035407133e+02, 0., 0., 1.);
    cv::Mat dist1 = (cv::Mat_<double>(5,1) <<
        5.9085906436082768e-02, -6.2062049999354461e-02,
       -4.2127875324890858e-03, -4.0428396364065671e-04, 0.);
    cv::Mat R1 =  (cv::Mat_<double>(3,3) <<
        9.9992050723396342e-01, -7.5619535204292680e-03,
       -1.0089403943158532e-02, 7.6080712339965124e-03,
       9.9996075032904486e-01, 4.5403803223814567e-03,
       1.0054673792410714e-02, -4.6167802988988796e-03,
       9.9993879256412488e-01);
    cv::Mat R2 = (cv::Mat_<double>(3,3) <<
        9.9990719408534878e-01, -1.0043377142692959e-02,
       9.2050959763823655e-03, 1.0085419948126504e-02,
       9.9993886915486030e-01, -4.5323567565640306e-03,
       -9.1590130928343328e-03, 4.6247733856343461e-03,
       9.9994736058969469e-01);
    cv::Mat P1 = (cv::Mat_<double>(3,4) <<
        4.9297293111444242e+02, 0., 3.2864122009277344e+02, 0., 0.,
       4.9297293111444242e+02, 2.1812635612487793e+02, 0., 0., 0., 1.,
       0.);
    cv::Mat P2 = (cv::Mat_<double>(3,4) <<
        4.9297293111444242e+02, 0., 3.2864122009277344e+02,
       -4.9681965181898590e+01, 0., 4.9297293111444242e+02,
       2.1812635612487793e+02, 0., 0., 0., 1., 0.);
    cv::Mat Q = (cv::Mat_<float>(4,4) <<
        1., 0., 0., -3.2864122009277344e+02, 0., 1., 0.,
       -2.1812635612487793e+02, 0., 0., 0., 4.9297293111444242e+02, 0.,
       0., 9.9225730968881827e+00, 0.);
    double baseline = 1.0077095772657289e-01;
    double focal = 4.9297293111444242e+02;
    cv::Mat cam0_map1, cam0_map2, cam1_map1, cam1_map2;
    initUndistortRectifyMap(cam0, dist0, R1, P1, cv::Size2i(640,400), CV_32FC1, cam0_map1, cam0_map2);
    initUndistortRectifyMap(cam1, dist1, R2, P2, cv::Size2i(640,400), CV_32FC1, cam1_map1, cam1_map2);
    cv::cuda::GpuMat g_cam0_map1(cam0_map1);
    cv::cuda::GpuMat g_cam0_map2(cam0_map2);
    cv::cuda::GpuMat g_cam1_map1(cam1_map1);
    cv::cuda::GpuMat g_cam1_map2(cam1_map2);

    //void *unified_ptr;
    //cudaMallocManaged(&unified_ptr, 1280*400);
    //cv::Mat frame(400, 1280, CV_8UC1, unified_ptr);
    //cv::cuda::GpuMat g_frame(400, 1280, CV_8UC1, unified_ptr);
    cv::Mat frame(400, 1280, CV_8UC1);
    cv::Mat frame_l, frame_r, frame_result;
    cv::cuda::GpuMat g_frame(400, 1280, CV_8UC1);

    /*void *unified_ptr_l, *unified_ptr_r;    
    cudaMallocManaged(&unified_ptr_l, 640*400);
    cv::cuda::GpuMat g_frame_l_rect(400,640, CV_8UC1, unified_ptr_l);
    cv::Mat left(400, 640, CV_8UC1, unified_ptr_l);
    cudaMallocManaged(&unified_ptr_r, 640*400);
    cv::cuda::GpuMat g_frame_r_rect(400,640, CV_8UC1, unified_ptr_r);
    cv::Mat right(400, 640, CV_8UC1, unified_ptr_r);*/

    /*void *unified_ptr_disp;
    cudaMallocManaged(&unified_ptr_disp, 640*400*2);
    cv::cuda::GpuMat g_disp_map(400, 640, CV_16S, unified_ptr_disp);
    cv::Mat disp_map(400, 640, CV_16S, unified_ptr_disp);*/
    //cv::cuda::GpuMat g_disp_map(400, 640, CV_16SC1);
    //cv::cuda::GpuMat g_disp_map(400,640,CV_8UC1);
    cv::cuda::GpuMat g_frame_l_rect;
    cv::cuda::GpuMat g_frame_r_rect;
    cv::cuda::GpuMat g_disp_map;
    cv::cuda::GpuMat g_disp_map_scaled;
    cv::cuda::GpuMat g_disp_map_filtered;
    //cv::cuda::GpuMat g_disp_map_scaled(400, 640, CV_32FC1);
    cv::cuda::GpuMat g_3d_img;
    cv::cuda::GpuMat g_split_img[3];
    cv::cuda::GpuMat g_depth_img;
    cv::cuda::GpuMat g_depth_img_th;
    cv::cuda::GpuMat g_depth_img_scaled;
    cv::cuda::GpuMat g_disp_map_c;
    cv::cuda::GpuMat g_frame_l_buf;
    cv::cuda::GpuMat g_frame_r_buf;
    cv::Mat disp_map;
    cv::Mat img_3d;

    //int ndisp, iters, levels, nr_plane;
    //cv::cuda::StereoConstantSpaceBP::estimateRecommendedParams(640, 400, ndisp, iters, levels, nr_plane);
    //cout <<  ndisp << "," << iters << "," << levels << "," <<  nr_plane << "\n";

    //cv::Ptr<cv::cuda::StereoSGM> sgm = cv::cuda::createStereoSGM(0, 128); //very bad result
    cv::Ptr<cv::cuda::StereoBM> bm = cv::cuda::createStereoBM(64, 19);
    //bm->setPreFilterType(cv::cuda::StereoBM::PREFILTER_XSOBEL);
    //bm->setPreFilterType(0);
    //bm->setUniquenessRatio(5);
    //bm->setTextureThreshold(4);
    //cv::Ptr<cv::cuda::DisparityBilateralFilter> filter = cv::cuda::createDisparityBilateralFilter(64);
    //cv::Ptr<cv::cuda::StereoConstantSpaceBP> csbp = cv::cuda::createStereoConstantSpaceBP(ndisp, iters, levels, nr_plane);
    //cv::Ptr<cv::cuda::StereoConstantSpaceBP> csbp = cv::cuda::createStereoConstantSpaceBP(128,8,4,4,CV_16SC1);
    //cout<<csbp->getMaxDataTerm()<<"\n";

    //cv::cuda::Stream cuda_stream(cudaStreamNonBlocking);
    cv::cuda::Stream cuda_stream1;
    cv::cuda::Stream cuda_stream2(cudaStreamNonBlocking);

    //cv::Ptr< cv::cuda::Filter > filter = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 1, 1, 1, 1, cv::BORDER_DEFAULT);

    cv::cuda::GpuMat g_frame_l, g_frame_r;
    cv::namedWindow("example", cv::WINDOW_AUTOSIZE);
    int n=0;
    char f_name[32];
    bool not_init=true;
    double min, max;
    bool working=false;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_a, t_b;
    while(1) {
        if (cap.grab()) {
            cap.retrieve(frame);            
            if (frame.empty()) {
                cout << "blank frame?!\n";
                break;
            }
#if 1
            //t_a = chrono::high_resolution_clock::now();
            g_frame.upload(frame, cuda_stream1);
            frame_l = frame.colRange(frame.cols / 2, frame.cols);
            frame_r = frame.colRange(0, frame.cols / 2);
            g_frame_l = g_frame.colRange(g_frame.cols / 2, g_frame.cols);
            g_frame_r = g_frame.colRange(0, g_frame.cols / 2);
            cv::cuda::remap(g_frame_l, g_frame_l_rect, g_cam0_map1, g_cam0_map2, INTER_LINEAR, BORDER_CONSTANT, cv::Scalar(), cuda_stream1);
            cv::cuda::remap(g_frame_r, g_frame_r_rect, g_cam1_map1, g_cam1_map2, INTER_LINEAR, BORDER_CONSTANT, cv::Scalar(), cuda_stream1);
            //t_b = chrono::high_resolution_clock::now();
            //cout<<"wrap:"<< chrono::duration_cast<chrono::milliseconds>(t_b-t_a).count()<<"ms"<<endl;
//            if (cuda_stream.queryIfComplete()) {
            //g_depth_img_scaled.download(disp_map);
            //if (!disp_map.empty) imshow("example", disp_map);
            n++;
            if (n==4) {
                n=0;
                working=true;
            //t_a = chrono::high_resolution_clock::now();
            g_frame_l_rect.copyTo(g_frame_l_buf, cuda_stream2);
            g_frame_r_rect.copyTo(g_frame_r_buf, cuda_stream2);
            bm->compute(g_frame_l_buf, g_frame_r_buf, g_disp_map, cuda_stream2);
            //filter->apply(g_disp_map, g_frame_l_buf, g_disp_map_filtered, cuda_stream2);
            //cv::cuda::reprojectImageTo3D(g_disp_map_filtered, g_3d_img, Q, 3, cuda_stream2);
            //cv::cuda::split(g_3d_img, g_split_img, cuda_stream2);
            //t_b = chrono::high_resolution_clock::now();
            //cout<<"disp:"<< chrono::duration_cast<chrono::milliseconds>(t_b-t_a).count()<<"ms"<<endl;
            //t_b = chrono::high_resolution_clock::now();
            //cout<<"csbp:"<< chrono::duration_cast<chrono::milliseconds>(t_b-t_a).count()<<"ms"<<endl;
            //filter->apply(g_disp_map, g_frame_l_rect, g_disp_map_filtered);
            //cv::cuda::minMax(g_disp_map, &min, &max);
            //cout << min << "," << max << "," << g_disp_map.type() << endl;
            //cv::cuda::multiply(g_disp_map, cv::Scalar(0.0625), g_disp_map2, 1,CV_32FC1);
            //g_disp_map.convertTo(g_disp_map_scaled, CV_32FC1, 1.0/16.0);
            //g_disp_map_scaled.convertTo(g_disp_map_8u, CV_8UC1, 255.0/63);
            //cuda_stream.waitForCompletion();
            //auto t_a = chrono::high_resolution_clock::now();

            //g_disp_map.convertTo(g_disp_map_scaled, CV_32FC1);
            //cv::cuda::minMax(g_depth_img, &min, &max);
            //cout << min << "," << max << "," << g_depth_img.type() << endl;

//            cv::cuda::reprojectImageTo3D(g_disp_map, g_3d_img, Q, 3);
            //auto t_b = chrono::high_resolution_clock::now();
            //cout<<"with cuda:"<< chrono::duration_cast<chrono::milliseconds>(t_b-t_a).count()<<"ms"<<endl;
//            cv::cuda::split(g_3d_img, g_split_img);
            //double min, max;
            //cv::cuda::minMax(g_split_img[2], &min, &max);
            //cout << min << "," << max << endl;
            //cv::cuda::resize(g_split_img[2], g_depth_img, cv::Size(320, 200), 0, 0);
//            cv::cuda::threshold(g_split_img[2], g_split_img[1], 0, 0, THRESH_TOZERO); //reuse unused
//            cv::cuda::threshold(g_split_img[1], g_split_img[0], 10 , 0, THRESH_TRUNC);
//            g_split_img[0].convertTo(g_depth_img, CV_16UC1, 1000);
            
            }
//            }
            //cv::cuda::normalize(g_disp_map_scaled, g_disp_map_8u, 0, 255, NORM_MINMAX, CV_8UC1);
            //filteri->apply(g_frame_l, g_frame_l);
            //right = frame.colRange(0, frame.cols / 2);
            //left = frame.colRange(frame.cols / 2, frame.cols);
            if (working && cuda_stream2.queryIfComplete()) {
            working=false;
            //cv::cuda::reprojectImageTo3D(g_disp_map, g_3d_img, Q, 3);
            //cv::cuda::split(g_3d_img, g_split_img);
            //cv::cuda::minMax(g_split_img[2], &min, &max);
            //cout << min << "," << max << "," << g_split_img[2].type() << endl;
            //cv::cuda::divide(cv::Scalar(baseline * focal), g_disp_map, g_depth_img, 1, CV_32FC1);
            //cv::cuda::threshold(g_depth_img, g_depth_img_th, 5, 0, THRESH_TRUNC);
            //cv::cuda::normalize(g_depth_img_th, g_depth_img_scaled, 0, 255, NORM_MINMAX, CV_8UC1, cv::noArray());
            //cv::cuda::threshold(g_split_img[2], g_split_img[0], 5 , 0, THRESH_TRUNC); //reuse unused
            //cv::cuda::normalize(g_split_img[0], g_depth_img_scaled, 0, 255, NORM_MINMAX, CV_8UC1, cv::noArray());
            //g_depth_img_scaled.download(disp_map);
            //g_3d_img.download(img_3d);
            //g_frame_l.download(disp_map);
            cv::cuda::drawColorDisp(g_disp_map, g_disp_map_c, 64);
            //g_disp_map_c.download(disp_map);
            //cout << img_3d.at<Vec3f>(200, 320) << endl;
            //t_a = chrono::high_resolution_clock::now();
            //cv::cuda::threshold(g_split_img[2], g_depth_img_th, 6 , 0, THRESH_TRUNC);
            //cv::cuda::normalize(g_depth_img_th, g_depth_img_scaled, 0, 255, NORM_MINMAX, CV_8UC1, cv::noArray());
            //g_depth_img_th.convertTo(g_depth_img_scaled, CV_8UC1, 255.0/6.0);
            //g_depth_img_scaled.download(disp_map);
            //g_disp_map_filtered.convertTo(g_disp_map, CV_8UC1, 4);
            g_disp_map_c.download(disp_map);
            //t_b = chrono::high_resolution_clock::now();
            //cout<<"post:"<< chrono::duration_cast<chrono::milliseconds>(t_b-t_a).count()<<"ms"<<endl;
            //cv::hconcat(frame_l, disp_map, frame_result);
            imshow("example", disp_map);
            }
#else
            imshow("example", frame);
#endif
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
            imwrite("ll.png", frame_l);
            imwrite("rr.png", frame_r);
            imwrite("opencv_bm.png", disp_map);
            //imwrite("left"+std::to_string(n)+".png", left);
            //imwrite("right"+std::to_string(n)+".png", right);
            //n++;
        }
    }

    cap.release();
    //cudaFree(&unified_ptr);
    //cudaFree(&unified_ptr_l);
    //cudaFree(&unified_ptr_r);
    //cudaFree(&unified_ptr_disp);
    destroyAllWindows();

    return 0;
}
