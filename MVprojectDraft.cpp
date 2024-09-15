//============================================================================
// Name        : LicensePlateIdentificationProject.cpp
// Author      : Ian Wang
// Version     :
// Copyright   : Your copyright notice
// Description : License Plate Identification Project in C++, Ansi-style
//============================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>
#include <sstream>
#include <thread>
//#include <onnxruntime_cxx_api.h>
//#include <cpu_provider_factory.h>

using namespace std;
using namespace cv;

vector<Point2f> order_points(const vector<Point2f>& pts) {
    vector<Point2f> rect(4);

    // Find the top-left point which has the smallest sum (x + y)
    // Find the bottom-right point which has the largest sum (x + y)
    // Find the top-right point which has the smallest difference (x - y)
    // Find the bottom-left point which has the largest difference (x - y)

    float min_sum = numeric_limits<float>::max();
    float max_sum = numeric_limits<float>::lowest();
    float min_diff = numeric_limits<float>::max();
    float max_diff = numeric_limits<float>::lowest();

    for (const auto& pt : pts) {
        float sum = pt.x + pt.y;
        float diff = pt.x - pt.y;

        if (sum < min_sum) {
            min_sum = sum;
            rect[0] = pt; // top-left
        }
        if (sum > max_sum) {
            max_sum = sum;
            rect[2] = pt; // bottom-right
        }
        if (diff < min_diff) {
            min_diff = diff;
            rect[1] = pt; // top-right
        }
        if (diff > max_diff) {
            max_diff = diff;
            rect[3] = pt; // bottom-left
        }
    }

    return rect;
}

Mat four_point_transform(const Mat& image, const vector<Point2f>& pts) {
    // Order the points and unpack them individually
    auto rect = order_points(pts);
    Point2f tl = rect[0];
    Point2f tr = rect[1];
    Point2f br = rect[2];
    Point2f bl = rect[3];

    // Compute the width of the new image
    float widthA = sqrt(pow(br.x - bl.x, 2) + pow(br.y - bl.y, 2));
    float widthB = sqrt(pow(tr.x - tl.x, 2) + pow(tr.y - tl.y, 2));
    int maxWidth = static_cast<int>(max(widthA, widthB));

    // Compute the height of the new image
    float heightA = sqrt(pow(tr.x - br.x, 2) + pow(tr.y - br.y, 2));
    float heightB = sqrt(pow(tl.x - bl.x, 2) + pow(tl.y - bl.y, 2));
    int maxHeight = static_cast<int>(max(heightA, heightB));

    // Set the destination points
    vector<Point2f> dst = {
        Point2f(0, 0),
        Point2f(maxWidth - 1, 0),
        Point2f(maxWidth - 1, maxHeight - 1),
        Point2f(0, maxHeight - 1)
    };

    // Compute the perspective transform matrix and apply it
    Mat M = getPerspectiveTransform(rect, dst);
    Mat warped;
    warpPerspective(image, warped, M, Size(maxWidth, maxHeight));

    return warped;
}

Mat rec_pre_processing(Mat img, Size size = Size(168, 48)) {
    float mean_value = 0.588;
    float std_value = 0.193;

    resize(img, img, size);

    img.convertTo(img, CV_32F); // convert to float32
    img = img / 255 - mean_value;
    img = img / std_value;
    printf("%d--%d--%d\n", img.rows, img.cols, img.channels());

    vector<Mat> inMatList;
    inMatList.push_back(img);
    Mat ret = dnn::blobFromImages(inMatList, 1.0, size, Scalar(0, 0, 0), true, false, CV_32F);

    return ret;
}

vector<string> initialize_string(void) {
    vector<string> vec_tmp;
    string str[] = {"#","京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖","闽","赣",
                    "鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新",
                    "学","警","港","澳","挂","使","领","民","航","危","0","1","2","3","4","5","6","7",
                    "8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T",
                    "U","V","W","X","Y","Z","险","品"};
    int num = sizeof(str) / sizeof(str[0]);
    for (int n = 0; n < num; n++) {
        vec_tmp.push_back(str[n]);
    }
    return vec_tmp;
}

/// @brief 解码函数
string decodePlate(vector<int> preds) {
    int pre = 0;
    vector<int> newPreds;
    for (int i = 0; i < preds.size(); i++) {
        if (preds[i] != 0 && preds[i] != pre) {
            newPreds.push_back(preds[i]);
        }
        pre = preds[i];
    }
    string plate = "";
    vector<string> vec_string = initialize_string();
    for (int i = 0; i < newPreds.size(); i++) {
        plate += vec_string[newPreds[i]];
        // printf("%s\n",vec_string[newPreds[i]]);
    }
    return plate;
}

/// @brief 获取最大索引函数（假设有该函数）
int getMaxIndex(const vector<float>& vec) {
    return max_element(vec.begin(), vec.end()) - vec.begin();
}


/// @brief 推理并获取结果
string get_plate_result(Mat img, dnn::Net& session_rec) {
    /* 对输入图片进行预处理 */
    Mat img_processed = rec_pre_processing(img);

    /* 运行 ONNX 模型 */
    vector<Mat> outputs;
    session_rec.setInput(img_processed);
    session_rec.forward(outputs);

    /* 获取输出 */
    Mat y_onnx = outputs[0];

    vector<vector<float>> vec_float;
    vec_float.resize(21);

    for (int i = 0; i < y_onnx.size[0]; ++i) {
        for (int j = 0; j < y_onnx.size[1]; ++j) {
            for (int k = 0; k < y_onnx.size[2]; ++k) {
                vec_float[j].push_back(y_onnx.at<float>(i, j, k));
            }
        }
    }

    vector<int> plate_no;
    plate_no.resize(21);
    plate_no.clear();
    for(int i = 0; i < 21; i++){
        auto num = getMaxIndex(vec_float[i]);
        plate_no.push_back(num);
    }

    string plate_num = decodePlate(plate_no);
    return plate_num;
}

int main() {
	dnn::Net rocNet = dnn::readNet("/home/yanlong12138/eclipse/MVprojectDraft/src/best_crnn.onnx");
	dnn::Net detNet = dnn::readNet("/home/yanlong12138/eclipse/MVprojectDraft/src/plate_detect.onnx");
	for(int i = 1; i <= 10; i++){
		ostringstream file_path, window_name, output_path;
		file_path<<"/mnt/hgfs/shared files/MVproject/input/"<<i<<".png";
		window_name<<"Image "<<i;
		output_path<<"/mnt/hgfs/shared files/MVproject/output/"<<i<<"_output.jpg";
		Mat image = imread(file_path.str());
		namedWindow(window_name.str(),WINDOW_NORMAL);
		imshow(window_name.str(),image);
		waitKey(0);
		cvtColor(image,image,COLOR_BGR2GRAY); //to gray
		Mat filtered, edged, warped;
		bilateralFilter(image,filtered,13,15,15); // noise removal
		Canny(filtered,edged, 80,130,3); //edge

		vector<Vec2f> lines;
		HoughLines(edged, lines, 1, CV_PI/ 180, 100);
		if (lines.empty()) {
		        std::cerr << "No lines detected!" << std::endl;
		        return -1;
		    }

		// 选择第一条直线
		float rho = lines[0][0], theta = lines[0][1];
		float a = cos(theta), b = sin(theta);
		float x0 = a * rho, y0 = b * rho;
		Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
		Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

		double angle = theta * 180 / CV_PI - 90;
		    if (angle < -45) {
		        angle += 90;
		    }

		 //获取图像中心
		Point2f center(edged.cols / 2.0, edged.rows / 2.0);

		// 计算旋转矩阵
		Mat rotMat = getRotationMatrix2D(center, angle, 1.0);

		// 旋转图像
		Mat rotated;
		warpAffine(filtered, rotated, rotMat, edged.size());
		Mat shold;
		threshold(rotated, shold, 0, 255, THRESH_OTSU + THRESH_BINARY);

//		warped = four_point_transform(edged, center);

		int minChangeCountRow = 10;
		int maxChangeCountRow = 100;
		vector<int> changeCounts;
		int changeCount;
		for (int i = 0; i < shold.rows; i++){
			for (int j = 0; j < shold.cols - 1; j++){
				int pixel_front = shold.at<char>(i, j);
				int pixel_back = shold.at<char>(i, j + 1);
				if (pixel_front != pixel_back){
					changeCount++;
				}
			}
			changeCounts.push_back(changeCount);
			changeCount = 0;
		}
		for (int i = 0; i < shold.rows; i++){
				if (changeCounts[i] < minChangeCountRow or changeCounts[i] > maxChangeCountRow){
					for (int j = 0; j < shold.cols; j++){
						shold.at<char>(i, j) = 0;
					}
				}
			}


		string plate_string = get_plate_result(image,detNet);
//		printf("plate:%s\n",plate_string.c_str());

		imshow(window_name.str(),shold);
		waitKey(0);
		destroyAllWindows();

//		imwrite(output_path.str(), rotated);
	}
	return 0;
}
