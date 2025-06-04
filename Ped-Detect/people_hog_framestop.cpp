/* Modifications by Nathaniel Martinez */

/*
TO Compile:
make clean && make -j 6

To run:
./people_hog_framestop --video=<Video File Name>.mp4 --store=<Logfile name by default it's named pedestrian_log>.txt
*/

// This is based on the example code on the opencv website, which can be found here: https://docs.opencv.org/4.5.4/df/d54/samples_2cpp_2peopledetect_8cpp-example.html
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

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates the use of the HoG descriptor.");
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

    cout << "Press 'q' or <ESC> to quit." << endl;

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    Mat frame;
    ofstream logFile(logFileName);
    vector<string> logBuffer;
    const int logBatchSize = 50;

    omp_set_num_threads(3);

    for (;;)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "Finished reading: empty frame" << endl;
            break;
        }

        int64 t = getTickCount();

        // Split the frame into smaller regions and process them in parallel
        vector<Rect> found; // Shared vector for final results
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

        t = getTickCount() - t;

        // Display FPS and detection mode
        {
            ostringstream buf;
            buf << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t);
            putText(frame, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
        }

        // Draw rectangles around detected pedestrians
        for (Rect &r : found)
        {
            r.x += cvRound(r.width * 0.1);
            r.width = cvRound(r.width * 0.8);
            r.y += cvRound(r.height * 0.07);
            r.height = cvRound(r.height * 0.8);
            rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
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
                logBuffer.clear(); // Clear the buffer after writing
                logFile.flush();   // Ensure the file is updated immediately
            }
        }

        imshow("People detector", frame);

        // Interact with user
        const char key = (char)waitKey(1);
        if (key == 27 || key == 'q') // ESC
        {
            cout << "Exit requested" << endl;
            break;
        }
    }

    // Write remaining logs on exit
    if (!logBuffer.empty()) {
        for (const string &entry : logBuffer) {
            logFile << entry;
        }
        logBuffer.clear();
    }

    return 0;
}
