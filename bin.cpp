#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/print.h>
#include <pcl/console/print.h>

#include <chrono>
#include <ctime>

#include "include/entropyOut.h"
#include "include/binSegmentation.h"
#include "include/pointposeOut.h"

//----------------------------------------------------------------------------- //
int main(int argc, char **argv)
{

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(argv[1], *source) == -1)
    {
        PCL_ERROR(" error opening file ");
        return (-1);
    }
    std::cout << "cloud orginal size: " << source->size() << std::endl;

    //time computation
    auto start = std::chrono::steady_clock::now();

    pcl::console::print_highlight("BinSegmentation...\n");
    //BIN SEGMENTATION -----------------------------------------------------------------------
    BinSegmentation bin;
    bin.setInputCloud(source);
    bin.setNumberLines(4);
    bin.setPaddingDistance(0.05); // 5cm from the bin walls
    bin.setMaxBinHeight(0.3);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_grasp(new pcl::PointCloud<pcl::PointXYZRGB>);
    bool bin_result = bin.compute(cloud_grasp);
    if (bin_result == false)
        return -1;

    pcl::ModelCoefficients::Ptr plane = bin.getPlaneGroundPoints();
    pcl::PointCloud<pcl::PointXYZ>::Ptr top_vertices = bin.getVerticesBinContour();

    //time computation
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "duration segmentation: " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;
    std::cout << std::endl;

    auto startE = std::chrono::steady_clock::now();

    pcl::console::print_highlight("EntropyFilter...\n");
    // ENTROPY FILTER -----------------------------------------------------------------------
    //
    EntropyFilter ef;
    ef.setInputCloud(cloud_grasp);
    ef.setVerticesBinContour(top_vertices);
    ef.setDownsampleLeafSize(0.005);
    ef.setEntropyThreshold(0.8);
    ef.setKLocalSearch(500);        // Nearest Neighbour Local Search
    ef.setCurvatureThreshold(0.01); //Curvature Threshold for the computation of Entropy
    ef.setDepthThreshold(0.03);
    ef.setAngleThresholdForConvexity(5);
    ef.useWeightedEntropy(true);
    ef.optimizeNumberOfClouds(true);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds_result;
    bool entropy_result = ef.compute(clouds_result);
    if (entropy_result == false)
        return -1;

    pcl::ModelCoefficients::Ptr plane_ef = ef.getReferencePlane();

    pcl::console::print_highlight("PointPose...\n");
    // GRASP POINT --------------------------------------------------------------------------
    PointPose pp;
    pp.setSourceCloud(source);
    pp.setInputVectorClouds(clouds_result);
    pp.setRefPlane(plane_ef);

    std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>> transformation_vector;
    std::vector<float> margin_values;
    int number = pp.compute(transformation_vector, margin_values);

    // CHOICE OF THE POSE TO USE AMONG THE SET CALCULATED ------------------------------------
    int max_id_margin = std::max_element(margin_values.begin(), margin_values.end()) - margin_values.begin();
    Eigen::Affine3d matrix = transformation_vector[max_id_margin];
    float margin = margin_values[max_id_margin];

    std::cout << std::endl;
    pcl::console::print_highlight("Done...\n");

    std::cout << "Transformation Matrxi: \n"
              << matrix.matrix() << std::endl;
    std::cout << "Margin available: " << margin << " meters" << std::endl;

    //time computation
    auto endE = std::chrono::steady_clock::now();
    auto diff2 = endE - startE;
    std::cout << "duration entropy filter: " << std::chrono::duration<double, std::milli>(diff2).count() << " ms" << std::endl;

    auto diffAll = endE - start;
    std::cout << "overall processing took: " << std::chrono::duration<double, std::milli>(diffAll).count() << " ms" << std::endl;

    bin.visualize(false, true, false);
    pp.visualizeCloudGrasp(cloud_grasp);
    ef.visualizeAll(false);

    return 0;
}
