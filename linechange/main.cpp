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
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA); //��ȣŸŶ �鿣�� ����
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); //��ȣŸŶ ����̽� ����
    auto output_names = net.getUnconnectedOutLayersNames(); //��� ���̾� �̸�

    //dxl_open();//���̳��ͼ� ���� ������ �ϱ����� �Լ� ȣ��
    VideoCapture cap(src, CAP_GSTREAMER);//ķ���κ��� ���� �޾ƿ� ->gstreamer�̿�
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

        cv::dnn::blobFromImage(frame, blob, 1 / 255.f, cv::Size(224, 224), cv::Scalar(), true, false, CV_32F); //���� �����������κ��� 4���� ��Ӱ�ü ���� �� ��ȯ
        net.setInput(blob); //��Ʈ��ũ �Է� ����
        net.forward(detections, output_names); //��Ʈ��ũ ������ ����(�߷�)

        std::vector<int> indices[NUM_CLASSES]; //��ü ������ ������ vecotr �迭
        std::vector<cv::Rect> boxes[NUM_CLASSES]; //��ü�� �ٿ�� �ڽ��� ������ vector �迭
        std::vector<float> scores[NUM_CLASSES]; //��ü ���� Ȯ���� ������ vector �迭

        for (auto& output : detections) //��� ���̾ ���� �ݺ�
        {
            const auto num_boxes = output.rows; //���� ��ü ���� ����
            for (int i = 0; i < num_boxes; i++) //�ݺ���
            {
                auto x = output.at<float>(i, 0) * frame.cols; //���� ��ü ���� �߽� x��ǥ ����
                auto y = output.at<float>(i, 1) * frame.rows; //���� ��ü ���� �߽� x��ǥ ����
                auto width = output.at<float>(i, 2) * frame.cols; //���� ��ü �ٿ�� �ڽ� �� ����
                auto height = output.at<float>(i, 3) * frame.rows; //���� ��ü �ٿ�� �ڽ� ���� ����
                cv::Rect rect(x - width / 2, y - height / 2, width, height); //���� ��ü ���� �簢�� ����
                for (int c = 0; c < NUM_CLASSES; c++) //Ŭ���� ������ŭ �ݺ�
                {
                    auto confidence = *output.ptr<float>(i, 5 + c); //���� Ȯ�� ����
                    if (confidence >= CONFIDENCE_THRESHOLD && rect.area() <= 4000) //�����س��� �Ӱ谪(0.9)���� ũ��
                    {
                        boxes[c].push_back(rect); //�ٿ�� �ڽ� ����
                        scores[c].push_back(confidence); //Ȯ�� ����

                    }
                }
            }
        }

        for (int c = 0; c < NUM_CLASSES; c++) //Ŭ���� ������ŭ �ݺ�
            cv::dnn::NMSBoxes(boxes[c], scores[c], 0.7, NMS_THRESHOLD, indices[c]); //������ Ȯ���� ���� ���͸�


        for (int c = 0; c < NUM_CLASSES; c++) //Ŭ���� ����ŭ �ݺ�
        {
            for (int i = 0; i < indices[c].size(); ++i) //���� ������ŭ �ݺ�
            {
                auto idx = indices[c][i]; //���� ��ü ��ȣ ����
                const auto& rect = boxes[c][idx]; //���� ��ü �ٿ�� �ڽ� ���� ����
                int myarea = rect.area(); //���� ��ü �ٿ�� �ڽ�
                if (myarea >= 1800 && myarea <= 4000) //�ٿ�� �ڽ� ������ ���ǿ� ������
                {
                    cout << "�簢�� ���� : " << rect.area() << endl; //���
                    change_signal = true; //���� ���� on
                }
                else change_signal = false; //�ƴҽ� ���� ���� ��ȣ off

            }
        }

        gray = My_cvtcolor(frame);

        dst = gray(Rect(0, gray.rows * 2 / 3, gray.cols, gray.rows / 3));
        side[0] = dst(Rect(0, 0, dst.cols / 3, dst.rows));
        side[1] = dst(Rect(dst.cols * 2 / 3, 0, dst.cols / 3, dst.rows));

        pt = My_line(dst, prev_pt);

        prev_pt = pt; //���� �����߽ɿ� ���� �����߽� ����

        frame_pt.x = pt.x; //���� �����߽� ����
        frame_pt.y = pt.y + gray.rows * 2 / 3; //���� �����߽� ����

        circle(frame, frame_pt, 2, Scalar(0, 0, 255), 2, -1); //�����߽� �׸���

        My_detect_line(side[0], side[1]);

        My_change_line(change_signal, prev_pt);

        change_signal = false;
        error = (dst.cols / 2 - pt.x); //�����߽� ���� ���� ����
        cout << "error : " << error << endl; //���
        //My_mtr(error);

        gettimeofday(&end1, NULL);
        gettimeofday(&start2, NULL);

        for (int c = 0; c < NUM_CLASSES; c++) //Ŭ���� ������ŭ �ݺ�
        {
            for (int i = 0; i < indices[c].size(); ++i) //���� ������ ���� �ݺ�
            {
                const auto color = colors[c % NUM_COLORS]; //�� ����

                auto idx = indices[c][i]; //���� ��ü ��ȣ ����
                const auto& rect = boxes[c][idx]; //���� ��ü �ٿ�� �ڽ� ���� ����
                //if(rect.area()>6000) continue;
                cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3); //ȭ�鿡 �簢�� �׸���

                std::string label_str = class_names[c] + ": " + cv::format("%.02lf", scores[c][idx]); //�νķ��� ������ string ��ü

                int baseline; //���� ����
                auto label_bg_sz = cv::getTextSize(label_str, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline); //���� ���� ����
                cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED); //���� �Է� �簢�� �׸���
                cv::putText(frame, label_str, cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0)); //���� �Է�
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

        diff1 = end1.tv_sec + end1.tv_usec / 1000000.0 - start1.tv_sec - start1.tv_usec / 1000000.0; //�ð� ����
        diff2 = end2.tv_sec + end2.tv_usec / 1000000.0 - start2.tv_sec - start2.tv_usec / 1000000.0; //�ð� ����
        cout << "diff1 : " << diff1 << endl;
        cout << "diff2 : " << diff2 << endl;

    }
    //dxl_close();
    return 0;
}

Mat My_cvtcolor(Mat frame)
{
    Mat gray; //���� ����
    cvtColor(frame, gray, COLOR_BGR2GRAY); //������ �ϳ��� �޾ƿ� GRAY�� ��ȯ
    medianBlur(gray, gray, 3); //�̵�� ���ͷ� ���͸�
    threshold(gray, gray, 170, 255, THRESH_BINARY); //����ȭ

    return gray; //��ȯ
}

Point2d My_line(Mat dst, Point2d prev)
{
    Mat labels, stats, centroids; //���� ����

    int cnt = connectedComponentsWithStats(dst, labels, stats, centroids); //���̺�
    Point2d pt; //��������
    vector<double> vecdou_distance; //��������
    double distance; //��������
    double* p; //��������
    int min_index; //��������
    if (cnt > 1) //���̺� ������ ����
    {
        for (int i = 1; i < cnt; i++) //�ݺ���
        {
            p = centroids.ptr<double>(i); //�����߽��� ����� �ּ� ����
            distance = abs(p[0] - prev.x); //x��ǥ�� ���� �����߽� �� ����
            vecdou_distance.push_back(distance); //vector�� ����
        }
        min_index = min_element(vecdou_distance.begin(), vecdou_distance.end()) - vecdou_distance.begin(); //�� �� ���� �ּ��� �ε��� ����
        pt = Point2d(centroids.at<double>(min_index + 1, 0), centroids.at<double>(min_index + 1, 1)); //�ش� �ε����� �����߽� ����
        if (abs(prev.x - pt.x) > 15 && dst.at<uchar>(pt.y, pt.x) != 255) pt = prev; //���� ���� �߽ɰ� ���� �����߽��� ���� 15���� ũ�� ���� �����߽��� ��ǥ�� 0�̸�
        vecdou_distance.clear(); //�ʱ�ȭ
    }
    else pt = prev; //���̺��� �� ���� ������ ���� �����߽� ����

    return pt; //��ȯ
}

void My_change_line(bool signal, Point2d& prev)
{
    if (signal == true) //��ȣ�� true��
    {
        cout << "on_change" << endl; //���
        if (spt[0].x == 1000) //���������� ���� ��
        {
            spt[1].x += 106; //������ ���� x��ǥ�� 128��ŭ �����̵�
            prev = spt[1]; //����
        }
        else //������ ������ ���� ��
        {
            prev = spt[0]; //���� ���� �����߽� ����
        }
    }
}

void My_mtr(double error)
{
    int L_speed = 60 - error / 3.7; //�ӵ�����
    int R_speed = 60 + error / 3.7; //�ӵ�����
    dxl_set_velocity(L_speed, -R_speed); //�ӵ� ���� �Լ�
}

void My_detect_line(Mat left, Mat right)
{
    Point2d pt[2]; //���� ����
    int left_index, right_index; //���� ����
    double left_area, right_area; //���� ����
    vector<double> left_minarea, right_minarea; //���� ����
    int left_cnt, right_cnt; //���� ����
    Mat left_labels, right_labels, left_stats, right_stats, left_centroids, right_centroids; //���� ����
    left_cnt = connectedComponentsWithStats(left, left_labels, left_stats, left_centroids); //���̺�
    right_cnt = connectedComponentsWithStats(right, right_labels, right_stats, right_centroids); //���̺�

    if (left_cnt > 1) { //���̺� �� ��ü�� 1������ ������
        for (int i = 1; i < left_cnt; i++) { //��ü ����ŭ �ݺ�
            int* p1 = left_stats.ptr<int>(i); //��ü�� ���̺� ������ ����
            left_area = abs(p1[4]); //���� ����
            left_minarea.push_back(left_area); //���Ϳ� ����
        }
        left_index = max_element(left_minarea.begin(), left_minarea.end()) - left_minarea.begin(); //�ִ� ���� ��ü�� �ּҹ�ȯ
        if (left_stats.ptr<int>(left_index + 1)[4] < 20) //�ִ� ������ 20���� ������
            pt[0] = Point(1000, 1000); //����
        else //�ƴ϶��
            pt[0] = Point2d(left_centroids.at<double>(left_index + 1, 0), left_centroids.at<double>(left_index + 1, 1)); //�ش� ��ü�� �����߽� ����
        left_minarea.clear(); //�ʱ�ȭ
    }
    else pt[0] = Point(1000, 1000); //���̺��� ���� �ʾҴٸ� ����

    if (right_cnt > 1) { //���̺� �� ��ü�� 1������ ������
        for (int i = 1; i < right_cnt; i++) { //��ü ����ŭ �ݺ�
            int* p1 = right_stats.ptr<int>(i); //��ü�� ���̺� ������ ����
            right_area = abs(p1[4]); //���� ����
            right_minarea.push_back(right_area); //���Ϳ� ����
        }
        right_index = max_element(right_minarea.begin(), right_minarea.end()) - right_minarea.begin(); //�ִ� ���� ��ü�� �ּ� ��ȯ
        if (right_stats.ptr<int>(right_index + 1)[4] < 20) //�ִ� ������ 20���� ������
            pt[1] = Point(1000, 1000); //����
        else //�ƴ϶��
            pt[1] = Point2d(right_centroids.at<double>(right_index + 1, 0), right_centroids.at<double>(right_index + 1, 1)); //�ش� ��ü �����߽� ����
        right_minarea.clear(); //�ʱ�ȭ
    }
    else pt[1] = Point(1000, 1000); //���̺��� ���� �ʾҴٸ� ����

    spt[0] = pt[0]; //����
    spt[1] = pt[1]; //����
}