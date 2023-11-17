#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
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

    Mat frame, left, right;
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
            right = frame.colRange(0, frame.cols / 2);
            left = frame.colRange(frame.cols / 2, frame.cols);
            imshow("example", frame);
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

    destroyAllWindows();

    return 0;
}
