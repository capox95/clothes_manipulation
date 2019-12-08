#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>

#include "include/findTarget.h"
#include "include/entropyOut.h"
#include "include/pointposeOut.h"
#include "include/wmSegmentation.h"

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::PointCloud<PointT> PointCloudIntT;

// Align a rigid object to a scene with clutter and occlusions
int main(int argc, char **argv)
{
    // Point clouds
    PointCloudT::Ptr object(new PointCloudT);
    PointCloudT::Ptr object_result(new PointCloudT);
    PointCloudT::Ptr scene(new PointCloudT);
    //PointCloudIntT::Ptr scene_map(new PointCloudIntT);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_seg(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Get input object and scene
    if (argc != 3)
    {
        pcl::console::print_error("Syntax is: %s object.pcd scene.pcd\n", argv[0]);
        return (1);
    }

    // Load object and scene
    pcl::console::print_highlight("Loading point clouds...\n");
    if (pcl::io::loadPCDFile<pcl::PointNormal>(argv[1], *object) < 0 || pcl::io::loadPCDFile<pcl::PointNormal>(argv[2], *scene) < 0)
    {
        pcl::console::print_error("Error loading object/scene file!\n");
        return (1);
    }

    pcl::console::print_highlight("Finding Washing Machine...\n");
    FindTarget ft;
    ft.object = object;
    ft.scene = scene;
    ft.apply_icp = false;
    ft.compute();

    object_result = ft.object_icp;

    pcl::console::print_highlight("Creating difference map...\n");
    Processing proc;
    proc.setSceneCloud(scene);
    proc.setObjectCloud(object_result);
    proc.compute(cloud_seg);
    proc.visualize();

    pcl::ModelCoefficients::Ptr plane_proc = proc.getPlaneUsed();

    pcl::console::print_highlight("EntropyFilter...\n");
    // Entropy Filter
    EntropyFilter ef;
    ef.setInputCloud(cloud_seg);
    ef.setDownsampleLeafSize(0.005);     // size of the leaf for downsampling the cloud, value in meters. Default = 5 mm
    ef.setEntropyThreshold(0.7);         // Segmentation performed for all points with normalized entropy value above this
    ef.setKLocalSearch(500);             // Nearest Neighbour Local Search
    ef.setCurvatureThreshold(0.01);      // Curvature Threshold for the computation of Entropy
    ef.setDepthThreshold(0.03);          // if the segment region has a value of depth lower than this -> not graspable (value in meters)
    ef.setAngleThresholdForConvexity(5); // convexity check performed if the angle btw two normal vectors is larger than this
    ef.setReferencePlane(plane_proc);
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
    pp.setSourceCloud(cloud_seg);
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

    //viz.spin();
    //ft.visualize();
    pp.visualizeCloudGrasp(cloud_seg);
    ef.visualizeAll(false);

    return (0);
}
