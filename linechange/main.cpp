#include <stdio.h>
#include <time.h>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <fstream>
#include "dxl.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
const float CONFIDENCE_THRESHOLD = 0.9;
const float NMS_THRESHOLD = 0.4;
const int NUM_CLASSES = 1;
// colors for bounding boxes
const cv::Scalar colors[] = {
{0, 255, 255},
{255, 255, 0},
{0, 255, 0},
{255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);
string src = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)192, height=(int)108, format=(string)NV12, framerate=(fraction)40/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)192,height=(int)108, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
Mat My_cvtcolor(Mat frame);
Point2d My_line(Mat dst, Point2d prev);
void My_change_line(bool signal, Point2d& prev);
void My_mtr(double error);
void My_detect_line(Mat left, Mat right);
Point2d spt[2];
int main()
{
    ifstream fp("robot.names");
    if (!fp.is_open()) { cerr << "Class file load failed!" << endl; return -1; }
    vector<String> class_names;
    string name;
    while (!fp.eof())
    {
        getline(fp, name);
        class_names.push_back(name);
    }
    fp.close();
    auto net = cv::dnn::readNetFromDarknet("kim.cfg", "kim_final.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA); //선호타킷 백엔드 설정
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); //선호타킷 디바이스 설정
    auto output_names = net.getUnconnectedOutLayersNames(); //출력 레이어 이름

    dxl_open();//다이나믹셀 관련 설정을 하기위한 함수 호출
    VideoCapture cap(src, CAP_GSTREAMER);//캠으로부터 영상 받아옴 ->gstreamer이용
    if (!cap.isOpened())
    {
        cerr << "Camera open failed!" << endl;
        return -1;
    }

    Mat frame, dst, gray, blob, side[2];
    Point2d pt, prev_pt(80, 180), frame_pt;
    double error;
    std::vector<cv::Mat> detections;
    int key;
    struct timeval start1, start2, end1, end2;
    double diff1, diff2;
    bool change_signal = false;
    while (true)
    {
        gettimeofday(&start1, NULL);
        cap >> frame;
        if (frame.empty())
        {
            cerr << "frame empty!!" << endl;
            break;
        }

        cv::dnn::blobFromImage(frame, blob, 1 / 255.f, cv::Size(224, 224), cv::Scalar(), true, false, CV_32F); //영상 한프레임으로부터 4차원 블롭객체 생성 후 반환
        net.setInput(blob); //네트워크 입력 설정
        net.forward(detections, output_names); //네트워크 순방향 실행(추론)

        std::vector<int> indices[NUM_CLASSES]; //객체 개수를 저장할 vecotr 배열
        std::vector<cv::Rect> boxes[NUM_CLASSES]; //객체의 바운딩 박스를 저장할 vector 배열
        std::vector<float> scores[NUM_CLASSES]; //객체 예측 확률을 저장할 vector 배열

        for (auto& output : detections) //출력 레이어에 따라 반복
        {
            const auto num_boxes = output.rows; //검출 객체 개수 저장
            for (int i = 0; i < num_boxes; i++) //반복문
            {
                auto x = output.at<float>(i, 0) * frame.cols; //검출 객체 영역 중심 x좌표 저장
                auto y = output.at<float>(i, 1) * frame.rows; //검출 객체 영역 중심 x좌표 저장
                auto width = output.at<float>(i, 2) * frame.cols; //검출 객체 바운딩 박스 폭 저장
                auto height = output.at<float>(i, 3) * frame.rows; //검출 객체 바운딩 박스 높이 저장
                cv::Rect rect(x - width / 2, y - height / 2, width, height); //검출 객체 영역 사각형 생성
                for (int c = 0; c < NUM_CLASSES; c++) //클래스 갯수만큼 반복
                {
                    auto confidence = *output.ptr<float>(i, 5 + c); //검출 확률 저장
                    if (confidence >= CONFIDENCE_THRESHOLD && rect.area() <= 4000) //지정해놓은 임계값(0.9)보다 크면
                    {
                        boxes[c].push_back(rect); //바운딩 박스 저장
                        scores[c].push_back(confidence); //확률 저장

                    }
                }
            }
        }

        for (int c = 0; c < NUM_CLASSES; c++) //클래수 갯수만큼 반복
            cv::dnn::NMSBoxes(boxes[c], scores[c], 0.7, NMS_THRESHOLD, indices[c]); //지정한 확률로 정보 필터링


        for (int c = 0; c < NUM_CLASSES; c++) //클래스 수만큼 반복
        {
            for (int i = 0; i < indices[c].size(); ++i) //검출 개수만큼 반복
            {
                auto idx = indices[c][i]; //검출 객체 번호 저장
                const auto& rect = boxes[c][idx]; //검출 객체 바운딩 박스 영역 저장
                int myarea = rect.area(); //검출 객체 바운딩 박스
                if (myarea >= 1800 && myarea <= 4000) //바운딩 박스 영역이 조건에 맞으면
                {
                    cout << "사각형 넓이 : " << rect.area() << endl; //출력
                    change_signal = true; //차선 변경 on
                }
                else change_signal = false; //아닐시 차선 변경 신호 off

            }
        }

        gray = My_cvtcolor(frame);

        dst = gray(Rect(0, gray.rows * 2 / 3, gray.cols, gray.rows / 3));
        side[0] = dst(Rect(0, 0, dst.cols / 3, dst.rows));
        side[1] = dst(Rect(dst.cols * 2 / 3, 0, dst.cols / 3, dst.rows));

        pt = My_line(dst, prev_pt);

        prev_pt = pt; //이전 무게중심에 현재 무게중심 저장

        frame_pt.x = pt.x; //현재 무게중심 대입
        frame_pt.y = pt.y + gray.rows * 2 / 3; //현재 무게중심 대입

        circle(frame, frame_pt, 2, Scalar(0, 0, 255), 2, -1); //무게중심 그리기

        My_detect_line(side[0], side[1]);

        My_change_line(change_signal, prev_pt);

        change_signal = false;
        error = (dst.cols / 2 - pt.x); //무게중심 차에 의한 오차
        cout << "error : " << error << endl; //출력
        My_mtr(error);

        gettimeofday(&end1, NULL);
        gettimeofday(&start2, NULL);

        for (int c = 0; c < NUM_CLASSES; c++) //클래수 갯수만큼 반복
        {
            for (int i = 0; i < indices[c].size(); ++i) //검출 갯수에 따라 반복
            {
                const auto color = colors[c % NUM_COLORS]; //색 지정

                auto idx = indices[c][i]; //검출 객체 번호 저장
                const auto& rect = boxes[c][idx]; //검출 객체 바운딩 박스 영역 저장
                //if(rect.area()>6000) continue;
                cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3); //화면에 사각형 그리기

                std::string label_str = class_names[c] + ": " + cv::format("%.02lf", scores[c][idx]); //인식률을 저장할 string 객체

                int baseline; //글자 길이
                auto label_bg_sz = cv::getTextSize(label_str, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline); //글자 길이 저장
                cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED); //글자 입력 사각형 그리기
                cv::putText(frame, label_str, cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0)); //글자 입력
            }
        }
        // cvtColor(dst, dst, COLOR_GRAY2BGR);
        // cvtColor(side[0], side[0], COLOR_GRAY2BGR);
        // cvtColor(side[1], side[1], COLOR_GRAY2BGR);
        // circle(dst, pt, 2, Scalar(0, 0, 255), 2);
        // circle(side[0], spt[0], 2, Scalar(0, 0, 255), 2);
        // circle(side[1], spt[1], 2, Scalar(0, 0, 255), 2);
        imshow("frame", frame);
        //imshow("gray",gray);
        //imshow("dst",dst);
        //imshow("side0",side[0]);
        //imshow("side1",side[1]);
        key = waitKey(2);
        gettimeofday(&end2, NULL);
        if (key == 27) break;

        diff1 = end1.tv_sec + end1.tv_usec / 1000000.0 - start1.tv_sec - start1.tv_usec / 1000000.0; //시간 측정
        diff2 = end2.tv_sec + end2.tv_usec / 1000000.0 - start2.tv_sec - start2.tv_usec / 1000000.0; //시간 측정
        cout << "diff1 : " << diff1 << endl;
        cout << "diff2 : " << diff2 << endl;

    }
    dxl_close();
    return 0;
}

Mat My_cvtcolor(Mat frame)
{
    Mat gray; //변수 선언
    cvtColor(frame, gray, COLOR_BGR2GRAY); //프레임 하나를 받아와 GRAY로 변환
    medianBlur(gray, gray, 3); //미디언 필터로 필터링
    threshold(gray, gray, 170, 255, THRESH_BINARY); //이진화

    return gray; //반환
}

Point2d My_line(Mat dst, Point2d prev)
{
    Mat labels, stats, centroids; //변수 선언

    int cnt = connectedComponentsWithStats(dst, labels, stats, centroids); //레이블링
    Point2d pt; //변수선언
    vector<double> vecdou_distance; //변수선언
    double distance; //변수선언
    double* p; //변수선언
    int min_index; //변수선언
    if (cnt > 1) //레이블링 개수에 따라
    {
        for (int i = 1; i < cnt; i++) //반복문
        {
            p = centroids.ptr<double>(i); //무게중심이 저장된 주소 추출
            distance = abs(p[0] - prev.x); //x좌표와 이전 무게중심 차 저장
            vecdou_distance.push_back(distance); //vector에 저장
        }
        min_index = min_element(vecdou_distance.begin(), vecdou_distance.end()) - vecdou_distance.begin(); //그 중 차가 최소인 인덱스 저장
        pt = Point2d(centroids.at<double>(min_index + 1, 0), centroids.at<double>(min_index + 1, 1)); //해당 인덱스의 무게중심 저장
        if (abs(prev.x - pt.x) > 15 && dst.at<uchar>(pt.y, pt.x) != 255) pt = prev; //이전 무게 중심과 현재 무게중심의 차가 15보다 크고 현재 무게중심의 좌표가 0이면
        vecdou_distance.clear(); //초기화
    }
    else pt = prev; //레이블링이 된 것이 없으면 이전 무게중심 저장

    return pt; //반환
}

void My_change_line(bool signal, Point2d& prev)
{
    if (signal == true) //신호가 true면
    {
        cout << "on_change" << endl; //출력
        if (spt[0].x == 1000) //왼쪽차선이 없을 때
        {
            spt[1].x += 106; //오른쪽 차선 x좌표에 128만큼 평행이동
            prev = spt[1]; //대입
        }
        else //오른쪽 차선이 없을 때
        {
            prev = spt[0]; //왼쪽 차선 무게중심 대입
        }
    }
}

void My_mtr(double error)
{
    int L_speed = 60 - error / 3.7; //속도대입
    int R_speed = 60 + error / 3.7; //속도대입
    dxl_set_velocity(L_speed, -R_speed); //속도 제어 함수
}

void My_detect_line(Mat left, Mat right)
{
    Point2d pt[2]; //변수 선언
    int left_index, right_index; //변수 선언
    double left_area, right_area; //변수 선언
    vector<double> left_minarea, right_minarea; //변수 선언
    int left_cnt, right_cnt; //변수 선언
    Mat left_labels, right_labels, left_stats, right_stats, left_centroids, right_centroids; //변수 선언
    left_cnt = connectedComponentsWithStats(left, left_labels, left_stats, left_centroids); //레이블링
    right_cnt = connectedComponentsWithStats(right, right_labels, right_stats, right_centroids); //레이블링

    if (left_cnt > 1) { //레이블링 된 객체가 1개보다 많으면
        for (int i = 1; i < left_cnt; i++) { //객체 수만큼 반복
            int* p1 = left_stats.ptr<int>(i); //객체의 레이블링 데이터 저장
            left_area = abs(p1[4]); //면적 대입
            left_minarea.push_back(left_area); //벡터에 저장
        }
        left_index = max_element(left_minarea.begin(), left_minarea.end()) - left_minarea.begin(); //최대 면적 객체의 주소반환
        if (left_stats.ptr<int>(left_index + 1)[4] < 20) //최대 면적이 20보다 작으면
            pt[0] = Point(1000, 1000); //대입
        else //아니라면
            pt[0] = Point2d(left_centroids.at<double>(left_index + 1, 0), left_centroids.at<double>(left_index + 1, 1)); //해당 객체의 무게중심 대입
        left_minarea.clear(); //초기화
    }
    else pt[0] = Point(1000, 1000); //레이블링이 되지 않았다면 대입

    if (right_cnt > 1) { //레이블링 된 객체가 1개보다 많으면
        for (int i = 1; i < right_cnt; i++) { //객체 수만큼 반복
            int* p1 = right_stats.ptr<int>(i); //객체의 레이블링 데이터 저장
            right_area = abs(p1[4]); //면적 대입
            right_minarea.push_back(right_area); //벡터에 저장
        }
        right_index = max_element(right_minarea.begin(), right_minarea.end()) - right_minarea.begin(); //최대 면적 객체의 주소 반환
        if (right_stats.ptr<int>(right_index + 1)[4] < 20) //최대 면적이 20보다 작으면
            pt[1] = Point(1000, 1000); //대입
        else //아니라면
            pt[1] = Point2d(right_centroids.at<double>(right_index + 1, 0), right_centroids.at<double>(right_index + 1, 1)); //해당 객체 무게중심 대입
        right_minarea.clear(); //초기화
    }
    else pt[1] = Point(1000, 1000); //레이블링이 되지 않았다면 대입

    spt[0] = pt[0]; //대입
    spt[1] = pt[1]; //대입
}
