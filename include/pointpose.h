#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/StdVector>

typedef std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>> Affine3dVector;
typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> Vector3fVector;

struct CoordinateFramePoints
{
    pcl::PointXYZ x;
    pcl::PointXYZ y;
    pcl::PointXYZ z;
    pcl::PointXYZ o;
};

class PointPoseDrum
{

private:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_source;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud_grasp, m_cloud_projected;

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> m_clouds_vector;
    std::vector<CoordinateFramePoints> m_cfp;
    std::vector<CoordinateFramePoints> m_cfp_viz;

    pcl::ModelCoefficients m_axis, m_plane;
    std::vector<pcl::PointXYZ> m_pointOnAxis;

    Eigen::Vector3f m_trans;
    Eigen::Quaternionf m_rot;

    pcl::PointXYZ m_origin;
    pcl::ModelCoefficients m_line;

    Eigen::Vector3f _directionX, _directionY, _directionZ, centroid_centroid;

public:
    PointPoseDrum() : m_source(new pcl::PointCloud<pcl::PointXYZRGB>),
                      m_cloud_grasp(new pcl::PointCloud<pcl::PointXYZ>),
                      m_cloud_projected(new pcl::PointCloud<pcl::PointXYZ>)

    {
    }

    void setSourceCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in);

    void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

    void setDrumAxis(pcl::ModelCoefficients &axis);

    void setInputVectorClouds(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clouds);

    int compute(Vector3fVector &pointsOnAxis, Affine3dVector &transformation_matrix_vector);

    bool computeGraspPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                           Eigen::Vector3f &point, Eigen::Affine3d &transformation_matrix);

    void visualizeGrasp();

private:
    std::vector<int> orderEigenvalues(Eigen::Vector3f eigenValuesPCA);

    void getCoordinateFrame(Eigen::Vector3f &centroid, Eigen::Matrix3f &rotation, pcl::PointXYZ &pointOnTheLine,
                            Eigen::Vector3f &directionX, Eigen::Vector3f &directionZ);

    Eigen::Affine3d computeTransformation(Eigen::Vector3f &centroid, Eigen::Vector3f &directionX, Eigen::Vector3f &directionZ);

    void computeRefPlane(pcl::ModelCoefficients &axis, Eigen::Vector4f &centroid,
                         pcl::ModelCoefficients &plane, pcl::PointXYZ &pointOnTheLine);

    void projectPointsOntoPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::ModelCoefficients &plane,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_projected);

    void computeCoordinateFramePointsViz(Eigen::Vector3f &centroid, Eigen::Matrix3f &rotation, bool reverse);
};