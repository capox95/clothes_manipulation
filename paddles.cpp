#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "include/findTarget.h"
#include "include/drumModel.h"

/*
for dataset with samples paddle1.pcd <-> paddle9.pcd
processing time btw ~800 ms up to almost 3000 ms for difficult samples (paddle7 and paddle9)
*/

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;

// Align a rigid object to a scene with clutter and occlusions
int main(int argc, char **argv)
{
    //----------------------------------
    Eigen::Vector3f axis_pt, axis_dir;
    axis_pt << 0, -0.07, 0.07;
    axis_dir << 0, 0.707107, 0.707107;

    float drumCenterDistance = 0.4;
    float drumRadius = 0.25;
    //---------------------------------

    // Point clouds
    PointCloudT::Ptr object(new PointCloudT);
    PointCloudT::Ptr scene(new PointCloudT);
    PointCloudT::Ptr object_result(new PointCloudT);

    // Get input object and scene
    if (argc != 3)
    {
        pcl::console::print_error("Syntax is: %s object.pcd scene.pcd\n", argv[0]);
        return (1);
    }

    // Load object and scene
    pcl::console::print_highlight("Loading point clouds...\n");
    if (pcl::io::loadPCDFile<pcl::PointNormal>(argv[1], *object) < 0 ||
        pcl::io::loadPCDFile<pcl::PointNormal>(argv[2], *scene) < 0)
    {
        pcl::console::print_error("Error loading object/scene file!\n");
        return (1);
    }
    FindTarget ft;
    ft.object = object;
    ft.scene = scene;
    bool success = ft.compute();
    if (!success)
        return -1;

    object_result = ft.object_icp;

    DrumModel dm;
    dm.setInputCloud(object_result);
    dm.setDrumAxis(axis_pt, axis_dir);
    dm.setDrumCenterDistance(drumCenterDistance);
    dm.setDrumRadius(drumRadius);

    std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>> transformation_vector;
    dm.compute(transformation_vector);

    for (int i = 0; i < transformation_vector.size(); i++)
        std::cout << transformation_vector[i].matrix() << std::endl;

    ft.visualize();
    dm.visualize(scene);

    return (0);
}
