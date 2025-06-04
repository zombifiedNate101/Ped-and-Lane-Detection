/* Modifications by Nathaniel Martinez */

/*
TO Compile:
make clean && make -j 6

To run:
./lanesnpeds --video=<Video File Name>.mp4 --store=<Logfile name by default it's named pedestrian_log>.txt
*/

// This example code can be found here: https://docs.opencv.org/4.5.4/df/d54/samples_2cpp_2peopledetect_8cpp-example.html
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>

using namespace cv;
using namespace std;

static const string keys = "{ help h   |   | print help message }"
                           "{ camera c | 0 | capture video from camera (device index starting from 0) }"
                           "{ video v  |   | use video as input }"
                           "{ store   | pedestrian_log.txt | Log file name for pedestrian detection data }";

enum Mode { LANE, PED, BOTH };

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Merged Lane and Pedestrian Detection. Press 'l' for lanes, 'p' for pedestrians, 'b' for both, 'q'/ESC to quit.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int camera = parser.get<int>("camera");
    string file = parser.get<string>("video");
    string logFileName = parser.get<string>("store");

    if (logFileName.empty() || logFileName == "true") {
        logFileName = "pedestrian_log.txt";
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    VideoCapture cap;
    if (file.empty())
        cap.open(camera);
    else
    {
        file = samples::findFileOrKeep(file);
        cap.open(file);
    }
    if (!cap.isOpened())
    {
        cout << "Can not open video stream: '" << (file.empty() ? "<camera>" : file) << "'" << endl;
        return 2;
    }

    cout << "Press 'l' for lanes, 'p' for pedestrians, 'b' for both, 'q' or <ESC> to quit." << endl;

    
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    Mat frame;
    ofstream logFile(logFileName);
    vector<string> logBuffer;
    const int logBatchSize = 50;

    omp_set_num_threads(3);

    // Start with both enabled
    Mode mode = BOTH; 

    double startTime = (double)getTickCount();
    int frameCount = 0;
    double fps = 0.0;

    for (;;)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "Finished reading: empty frame" << endl;
            break;
        }

        int64 t = getTickCount();

        Mat displayFrame = frame.clone();

        // For pedestrian detection
        vector<Rect> found; 

        // Lane Detection
        if (mode == LANE || mode == BOTH)
        {
            Mat dst, cdstP;
            // Edge detection
            Mat blurred;
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    //cvtColor(dst, cdstP, COLOR_GRAY2BGR);
                }
                #pragma omp section
                {
                    // GaussianBlur(frame, blurred, Size(5, 5), 1.5);
                }
                #pragma omp section
                {
                    Canny(frame, dst, 70, 200, 3);
                }

            }

            Rect roi(frame.cols * 0, frame.rows * 0.5, frame.cols * 1, frame.rows * 0.25);
            Mat lowerHalf = dst(roi);

            vector<Vec4i> linesP;
            HoughLinesP(lowerHalf, linesP, 1, CV_PI/180, 80, 50, 5);

            #pragma omp parallel for
            for (size_t i = 0; i < linesP.size(); i++)
            {
                Vec4i l = linesP[i];
                Point pt1(l[0], l[1] + dst.rows * 0.5);
                Point pt2(l[2], l[3] + dst.rows * 0.5);

                double angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x) * 180.0 / CV_PI;
                if(abs(angle) < 25 || abs(angle) > 165) continue;

                #pragma omp critical
                {
                    line(displayFrame, pt1, pt2, Scalar(0,255,0), 3, LINE_AA);
                }
            }
        }

        // Pedestrian Detection
        if (mode == PED || mode == BOTH)
        {
            // Split the frame into smaller regions and process them in parallel
            #pragma omp parallel
            {
                // Each thread will have its own local vector to store results
                vector<Rect>* localFound = new vector<Rect>();
                #pragma omp for nowait
                for (int y = 0; y < frame.rows; y += frame.rows / omp_get_num_threads())
                {
                    Rect region(0, y, frame.cols, frame.rows / omp_get_num_threads());
                    Mat subFrame = frame(region);

                    Mat resizedFrame;
                    resize(subFrame, resizedFrame, Size(subFrame.cols / 2, subFrame.rows / 2));

                    // Detect pedestrians in the resized frame
                    hog.detectMultiScale(resizedFrame, *localFound, 0, Size(8, 8), Size(), 1.05, 2, false);

                    for (Rect &r : *localFound)
                    {
                        r.x *= 2;
                        r.y *= 2;
                        r.width *= 2;
                        r.height *= 2;
                    }
                    for (Rect &r : *localFound)
                    {
                        r.y += y;
                    }
                }
                #pragma omp critical
                found.insert(found.end(), localFound->begin(), localFound->end());
                delete localFound;
            }

            // Draw rectangles around detected pedestrians
            for (Rect &r : found)
            {
                r.x += cvRound(r.width * 0.1);
                r.width = cvRound(r.width * 0.8);
                r.y += cvRound(r.height * 0.07);
                r.height = cvRound(r.height * 0.8);
                rectangle(displayFrame, r.tl(), r.br(), cv::Scalar(0, 0, 255), 2);
            }

            // Add log data to the buffer
            if (!found.empty()) {
                ostringstream logEntry;
                logEntry << "Frame " << cap.get(CAP_PROP_POS_FRAMES) << ": " << found.size() << " pedestrians detected\n";
                for (Rect &r : found) {
                    logEntry << "Coordinates: " << r.x << ", " << r.y << ", " << r.width << ", " << r.height << endl;
                }
                logBuffer.push_back(logEntry.str());

                // Write to file in batches
                if (logBuffer.size() >= logBatchSize) {
                    for (const string &entry : logBuffer) {
                        logFile << entry;
                    }
                    logBuffer.clear();
                    logFile.flush();
                }
            }
        }

        // Calculate FPS
        frameCount++;
        double elapsedTime = ((double)getTickCount() - startTime) / getTickFrequency();
        if(elapsedTime >= 1.0){
            fps = frameCount / elapsedTime;
            frameCount = 0;
            startTime = (double)getTickCount();
        }

        // Display FPS and mode
        ostringstream buf;
        buf << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)(getTickCount() - t));
        string modeStr = (mode == LANE) ? "LANE" : (mode == PED) ? "PED" : "BOTH";
        putText(displayFrame, buf.str() + " Mode: " + modeStr, Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 0), 2, LINE_AA);

        imshow("Lane & Pedestrian Detector", displayFrame);

        // Interact with user
        const char key = (char)waitKey(1);
        if (key == 27 || key == 'q') // ESC
        {
            cout << "Exit requested" << endl;
            break;
        }
        else if (key == 'l')
        {
            mode = LANE;
            cout << "Switched to Lane Detection only." << endl;
        }
        else if (key == 'p')
        {
            mode = PED;
            cout << "Switched to Pedestrian Detection only." << endl;
        }
        else if (key == 'b')
        {
            mode = BOTH;
            cout << "Switched to BOTH Lane and Pedestrian Detection." << endl;
        }
    }

    // Write remaining logs on exit
    if (!logBuffer.empty()) {
        for (const string &entry : logBuffer) {
            logFile << entry;
        }
        logBuffer.clear();
    }

    destroyAllWindows();
    return 0;
}