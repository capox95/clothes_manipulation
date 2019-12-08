#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

typedef std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>> Affine3dVector;
typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> Vector3fVector;

struct CoordinateFramePoints
{
    pcl::PointXYZ x;
    pcl::PointXYZ y;
    pcl::PointXYZ z;
    pcl::PointXYZ o;
};

class PointPose
{

private:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_source;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud_grasp, m_cloud_projected;
    pcl::ModelCoefficients::Ptr m_plane;

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> m_clouds_vector;
    std::vector<CoordinateFramePoints> m_cfp;
    std::vector<CoordinateFramePoints> m_cfp_viz;

    Eigen::Vector3f m_trans;
    Eigen::Quaternionf m_rot;
    std::vector<pcl::PointXYZ> m_pointsCoordinateFrame;
    pcl::PointXYZ m_origin;
    pcl::ModelCoefficients m_line;

    Eigen::Vector3f _directionX, _directionY, _directionZ, _centroid;

public:
    PointPose() : m_source(new pcl::PointCloud<pcl::PointXYZRGB>),
                  m_cloud_grasp(new pcl::PointCloud<pcl::PointXYZ>),
                  m_cloud_projected(new pcl::PointCloud<pcl::PointXYZ>),
                  m_plane(new pcl::ModelCoefficients)

    {
    }

    void setInputVectorClouds(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clouds);

    void setSourceCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in);

    void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

    void setRefPlane(pcl::ModelCoefficients::Ptr &plane);

    Eigen::Vector3f getTranslation();

    Eigen::Quaternionf getRotation();

    Eigen::Vector3f getDirectionWrinkle();

    bool computeGraspPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                           Eigen::Affine3d &transformation_matrix, float &margin);

    int compute(Affine3dVector &transformation_matrix_vector, std::vector<float> &margins);

    void visualizeGrasp();

    void visualizeCloudGrasp(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in);

private:
    std::vector<int> orderEigenvalues(Eigen::Vector3f eigenValuesPCA);

    void getCoordinateFrame(Eigen::Vector3f &centroid, Eigen::Matrix3f &rotation,
                            Eigen::Vector3f &directionX, Eigen::Vector3f &directionZ);

    void computeCoordinateFramePointsViz(Eigen::Vector3f &centroid, Eigen::Matrix3f &rotation, bool reverse);

    Eigen::Affine3d computeTransformation(Eigen::Vector3f &centroid, Eigen::Vector3f &directionX, Eigen::Vector3f &directionZ);

    Eigen::Vector3f moveCentroid(Eigen::Vector4f centroid, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
};