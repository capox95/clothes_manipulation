#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/time.h>
#include <pcl/common/geometry.h>

#include "include/entropy.h"
#include "include/pointpose.h"
#include "include/segmentation.h"

/*
____________________________________________

sample1.pcd <---> sample13.pdc 
orientation: 0.9075 rad, 52 deg
line1: 0, 0.09, 0.11, 0, -0.787967, 0.615718;
line2: 0, 0.09, 0.11, 0, 0.615718, 0.787967;

data1.pcd <---> data4.pcd
orientation: 0.7592 rad, 43.5 deg
line1: 0, 0.04, 0.06, 0, -0.714004, 0.700142;
line2: 0, 0.04, 0.06, 0, 0.700142, 0.714004;
_____________________________________________

Performance:
sample1 <-> sample5, sample7: processing took less than 1000 ms (uses alternative approach)
sample6, sample8 <-> sample13: processing took up to 2500ms (sample9), 2000ms (sample8) (connected componets analysis)
*/

int main(int argc, char **argv)
{
    //----------------------------------------------------
    Eigen::Vector3f line1_pt, line1_dir, line2_pt, line2_dir;

    line1_pt << 0, 0.09, 0.11;
    line2_pt << 0, 0.09, 0.11;
    line1_dir << 0, -0.787967, 0.615718;
    line2_dir << 0, 0.615718, 0.787967;

    //line1_pt << 0, 0.04, 0.06;
    //line2_pt << 0, 0.04, 0.06;
    //line1_dir << 0, -0.714004, 0.700142;
    //line2_dir << 0, 0.700142, 0.714004;

    pcl::ModelCoefficients axis;
    axis.values = {line1_pt.x(), line1_pt.y(), line1_pt.z(), line1_dir.x(), line1_dir.y(), line1_dir.z()};

    float distanceCenter = 0.4;
    float radius = 0.25;
    float depth = 0.3;
    //----------------------------------------------------

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr segfilter_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(argv[1], *source) == -1)
    {
        PCL_ERROR(" error opening file ");
        return (-1);
    }
    std::cout << "cloud orginal size: " << source->size() << std::endl;

    SegFilter sf;
    sf.setSourceCloud(source);
    sf.setLine1(line1_pt, line1_dir);
    sf.setLine2(line2_pt, line2_dir);
    sf.setDistanceDrumCenter(distanceCenter);
    sf.setDrumDimensions(radius, depth);
    sf.compute(segfilter_cloud);

    EntropyFilterDrum ef;
    ef.setInputCloud(segfilter_cloud);
    ef.setEntropyThreshold(0.75);   // Segmentation performed for all points with normalized entropy value above this
    ef.setKLocalSearch(500);        // Nearest Neighbour Local Search
    ef.setCurvatureThreshold(0.03); // Curvature Threshold for the computation of Entropy
    ef.setDepthThreshold(0.02);     // 2 cm margin
    ef.setDrumRadius(radius);
    ef.setDrumAxis(axis);
    ef.optimizeNumberOfClouds(true);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds_result;
    if (ef.compute(clouds_result) == false)
        return -1;

    PointPoseDrum pp;
    pp.setSourceCloud(source);
    pp.setInputVectorClouds(clouds_result);
    pp.setDrumAxis(axis);

    std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>> transformation_vector;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points_vector;
    int number = pp.compute(points_vector, transformation_vector);

    // CHOICE OF THE POSE TO USE AMONG THE SET CALCULATED ------------------------------------
    std::vector<float> distances(points_vector.size());
    pcl::PointXYZ origin = {0, 0, 0};
    for (int i = 0; i < points_vector.size(); i++)
    {
        pcl::PointXYZ point = {points_vector[i].x(), points_vector[i].y(), points_vector[i].z()};
        distances[i] = pcl::geometry::distance(origin, point);
    }
    int max_id = std::min_element(distances.begin(), distances.end()) - distances.begin();
    Eigen::Affine3d matrix = transformation_vector[max_id];
    Eigen::Vector3f pointAxes = points_vector[max_id];

    std::cout << std::endl;
    pcl::console::print_highlight("Done...\n");

    std::cout << "Transformation Matrix: \n"
              << matrix.matrix() << std::endl;
    std::cout << "Point Axes : \n"
              << pointAxes << std::endl;

    sf.visualizeSeg();
    pp.visualizeGrasp();
    ef.visualizeAll(false);

    return 0;
}
