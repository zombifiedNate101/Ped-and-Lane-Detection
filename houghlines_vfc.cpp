#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "omp.h"

#define ESCAPE_KEY 27
#define SYSTEM_ERROR (-1)

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

    if(argc != 2)
    {
        cout << "Usage: " << argv[0] << " <video_file>" << endl;
        return -1;
    }

    string videoFile = argv[1];
    VideoCapture vcap(videoFile);
    namedWindow("Lane Detector");
    
    if(!vcap.isOpened()){
    	exit(SYSTEM_ERROR);
    }
    
    //vcap.set(CAP_PROP_FRAME_WIDTH, 640);
    //vcap.set(CAP_PROP_FRAME_HEIGHT, 640);
    
    namedWindow("Lane Detector", WINDOW_NORMAL);
    //namedWindow("Canny Tracing", WINDOW_NORMAL);

    
    resizeWindow("Lane Detector", 720, 720);
    //resizeWindow("Canny Tracing", 720, 720);


    
    Mat src, dst, cdstP, src_resized, cdstP_resized;
    
    //resize(src, src_resized, Size(720, 720));
    //resize(cdstP, cdstP_resized, Size(720, 720));
    
    double startTime = (double)getTickCount();
    int frameCount = 0;
    double fps = 0.0;
    
    while(1){
    	
	vcap >> src;

	if(src.empty())
	{
	    cout << "Error: Frame is empty." << endl;
	    break;
	}

	// Edge detection
	Mat blurred;
#pragma omp parallel sections
{
	#pragma omp section
	{
		//GaussianBlur(src, blurred, Size(5, 5), 1.5);//noise reduction; less necessary after filtering horizontal lines
	}
	#pragma omp section
	{
		Canny(src, dst, 70, 200, 3);
	}
	#pragma omp section
	{
		cvtColor(dst, cdstP, COLOR_GRAY2BGR);
	}
}
	
	Rect roi(src.cols * 0, src.rows * 0.5, src.cols * 1, src.rows * 0.25);	//restrict roi to lower half
	Mat lowerHalf = dst(roi);
	
	// Probabilistic Line Transform
	vector<Vec4i> linesP; // will hold the results of the detection
	
	

	HoughLinesP(lowerHalf, linesP, 1, CV_PI/180, 80, 50, 5 ); // runs the actual detection


	// Draw the lines
	#pragma omp parallel for
	for( size_t i = 0; i < linesP.size(); i++ )
	{
		Vec4i l = linesP[i];
		Point pt1(l[0], l[1] + dst.rows * 0.5);
		Point pt2(l[2], l[3] + dst.rows * 0.5);
		
		double angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x) * 180.0 / CV_PI;//filter out horizontal lines
		if(abs(angle) < 25 || abs(angle) > 165) continue;
		
		#pragma omp critical
		{
			line(src, pt1, pt2, Scalar(0,255,0), 3, LINE_AA);
			//line(cdstP, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
		}
	}
	
	
	//calculate FPS
        frameCount++;
        double elapsedTime = ((double)getTickCount() - startTime) / getTickFrequency();
        if(elapsedTime >= 1.0){
            fps = frameCount / elapsedTime;
            //reset the frame count and start time for the next second
            frameCount = 0;
            startTime = (double)getTickCount();
        }

        //show fps on the frame
        putText(src, "FPS: " + to_string((int)fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, LINE_AA);
        //putText(cdstP, "FPS: " + to_string((int)fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, LINE_AA);
        
        //Show results

        imshow("Lane Detector", src);
        //imshow("Canny Tracing", cdstP_resized);
        
	char winInput = waitKey(10);
	if(winInput == ESCAPE_KEY){break;}
    }

    destroyAllWindows();
    
    return 0;
}
