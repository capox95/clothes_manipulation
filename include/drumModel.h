#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/common/intersections.h>
#include <pcl/common/centroid.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>> Affine3dVector;

class DrumModel
{
public:
    pcl::PointCloud<pcl::PointNormal>::Ptr _input;

    pcl::PointXYZ _maxPaddlePoint, _minPaddlePoint, _centerPaddleProjected;
    pcl::PointXYZ _centerPaddlePoint, _centerPaddle2Point, _centerPaddle3Point, _centerPoint;
    std::vector<pcl::PointXYZ> _paddlesCenter;

    std::vector<pcl::ModelCoefficients> _planes;
    pcl::ModelCoefficients _line, _cylinder, _axis, _axisDrum;

    Eigen::Vector3f _origin, _axis_dir;

    float _paddleLength, _paddleHeight, _radius, _distanceCenter;

public:
    DrumModel() : _input(new pcl::PointCloud<pcl::PointNormal>)
    {
    }

    void setInputCloud(pcl::PointCloud<pcl::PointNormal>::Ptr &input);

    void setDrumAxis(Eigen::Vector3f &axis_pt, Eigen::Vector3f &axis_dir);

    void setDrumCenterDistance(float distance);

    void setDrumRadius(float radius);

    void visualize(pcl::PointCloud<pcl::PointNormal>::Ptr &scene, bool planes_flag = false,
                   bool cylinder_flag = false, bool lines_flag = false);

    void compute(Affine3dVector &transformation_matrix_vector);

    void findPlanes(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_plane, std::vector<pcl::ModelCoefficients> &planes);

    void estimateIntersactionLine(std::vector<pcl::ModelCoefficients> &planes, pcl::ModelCoefficients &line_model);

    void getPointsOnLine(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud, pcl::ModelCoefficients &line,
                         pcl::PointXYZ &maxFinPoint, pcl::PointXYZ &minFinPoint, pcl::PointXYZ &centerFinPoint);

    bool checkIfParallel(pcl::ModelCoefficients &line, pcl::ModelCoefficients &axis);

    pcl::PointXYZ projection(pcl::PointXYZ &point, pcl::ModelCoefficients &line);

    void calculateNewPoints(pcl::ModelCoefficients &cylinder, pcl::PointXYZ &centerCylinder,
                            pcl::PointXYZ &centerFin1, pcl::PointXYZ &centerFin2, pcl::PointXYZ &centerFin3);

    std::vector<pcl::PointXYZ> movePointsToPaddlesCenter(std::vector<pcl::PointXYZ> &points, pcl::PointXYZ &center, float height);

    float calculatePaddleHeight(pcl::PointCloud<pcl::PointNormal>::Ptr &input, pcl::ModelCoefficients &line, float &height);

    void computeTransformation(Affine3dVector &transformation_matrix_vector);

    //void computeDrumAxes(pcl::ModelCoefficients &line1, pcl::ModelCoefficients &line2);
    //void transformation(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out);
};
