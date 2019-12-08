#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::PointCloud<PointT> PointCloudIntT;

class Processing
{
private:
    pcl::ModelCoefficients::Ptr plane_;
    PointCloudT::Ptr scene_;
    PointCloudT::Ptr object_result_;
    PointCloudIntT::Ptr scene_map_, combined_hulls_;
    PointCloudT::Ptr cloud_hull_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_;

    Eigen::Vector4f centroid_;
    std::vector<pcl::ModelCoefficients> lines_;

    pcl::PointXYZ P1_, P2_; // intersection points
    pcl::PointXYZ P11_, P22_;

public:
    Processing() : plane_(new pcl::ModelCoefficients),
                   object_result_(new PointCloudT),
                   scene_(new PointCloudT),
                   scene_map_(new PointCloudIntT),
                   combined_hulls_(new PointCloudIntT),
                   cloud_hull_(new PointCloudT),
                   vertices_(new pcl::PointCloud<pcl::PointXYZ>)
    {
    }

    void compute(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_seg);

    void setSceneCloud(PointCloudT::Ptr scene);

    void setObjectCloud(PointCloudT::Ptr object);

    pcl::ModelCoefficients::Ptr getPlaneUsed();

    PointCloudIntT::Ptr getSceneMap();

    void visualize();

private:
    void
    getPointPlaneDistanceCloud(PointCloudT::Ptr &obj,
                               PointCloudT::Ptr &scene,
                               PointCloudT::Ptr scene_proj,
                               PointCloudIntT::Ptr &cloud_out,
                               pcl::ModelCoefficients::Ptr &plane);

    void computeDiffProjected(PointCloudT::Ptr obj,
                              PointCloudT::Ptr scene,
                              PointCloudIntT::Ptr cloud_output,
                              PointCloudT::Ptr hull,
                              pcl::ModelCoefficients::Ptr &plane);

    void segmentSceneCloud(PointCloudIntT::Ptr cloud,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_seg);

    void lineDetection(PointCloudT::Ptr &hull, std::vector<pcl::ModelCoefficients> &lines, pcl::PointXYZ &p1, pcl::PointXYZ &p2);

    PointCloudIntT::Ptr convexCrop(PointCloudT::Ptr obj, PointCloudIntT::Ptr cloud,
                                   pcl::PointXYZ &p1, pcl::PointXYZ &p2, pcl::ModelCoefficients::Ptr &plane);

    void updateVerticesCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr vertices,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr update,
                             pcl::ModelCoefficients::Ptr &plane);

    void croppingWithHull(pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_updated,
                          PointCloudIntT::Ptr cloud,
                          PointCloudIntT::Ptr hull);

    void combineHulls(PointCloudIntT::Ptr hull1, PointCloudIntT::Ptr hull2, PointCloudIntT::Ptr result);
};