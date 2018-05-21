#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <time.h>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = Mat( rows, cols, CV_32FC1 );
    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1 );
        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages = false )
{
    vector< String > files;
    glob( dirname, files );
    for ( size_t i = 0; i < files.size(); ++i )
    {
        Mat img = imread( files[i] ); // load the image
        if ( img.empty() )            // invalid image, skip it.
        {
            cout << files[i] << " is invalid!" << endl;
            continue;
        }
        if ( showImages )
        {
            imshow( "image", img );
            waitKey( 1 );
        }
        img_lst.push_back( img );
    }
}

void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size )
{
    Rect box;
    box.width = size.width;
    box.height = size.height;
    const int size_x = box.width;
    const int size_y = box.height;
    srand( (unsigned int)time( NULL ) );
    for ( size_t i = 0; i < full_neg_lst.size(); i++ )
        if ( full_neg_lst[i].cols >= box.width && full_neg_lst[i].rows >= box.height )
        {
            box.x = rand() % ( full_neg_lst[i].cols - size_x );
            box.y = rand() % ( full_neg_lst[i].rows - size_y );
            Mat roi = full_neg_lst[i]( box );
            neg_lst.push_back( roi.clone() );
        }
}

void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip )
{
    HOGDescriptor hog(wsize, Size( 10, 20 ), Size( 5, 10 ), Size( 10, 10), 9 );
    Mat gray;
    vector< float > descriptors;
    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
        if ( img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height )
        {
            Rect r = Rect(( img_lst[i].cols - wsize.width ) / 2,
                          ( img_lst[i].rows - wsize.height ) / 2,
                          wsize.width,
                          wsize.height);
            cvtColor( img_lst[i](r), gray, COLOR_BGR2GRAY );
            hog.compute( gray, descriptors);
            gradient_lst.push_back( Mat(descriptors).clone() ) ;
            if ( use_flip )
            {
                flip( gray, gray, 1 );
                hog.compute( gray, descriptors);
                gradient_lst.push_back(  Mat(descriptors).clone() );
            }
        }
    }
}


int main(int argc, char const *argv[])
{
    String pos_dir = "/data_train/Person";
    String neg_dir = "/data_train/NonePerson";
 
	Mat img = imread("test.jpg", IMREAD_COLOR);
	resize(img, img, Size(img.rows/2, img.cols/2));
	Size image_size = img.size();
	vector< Mat >  gradient_lst;
	vector< Mat > img_lst;
	Mat trainData;
	vector< Mat >  img_lst;

	img_lst.push_back(img);
	computeHOGs( image_size, img_lst, gradient_lst, false );

	convert_to_ml(gradient_lst, trainData);

    Ptr< SVM > svm = SVM::create();


    // svm->load("all_data.pkl");

    // cout<< svm->predict(trainData);

	return 0;
}