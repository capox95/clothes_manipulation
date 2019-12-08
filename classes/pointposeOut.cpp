#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/pca.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/visualization/cloud_viewer.h>

#include "../include/pointposeOut.h"

void PointPose::setInputVectorClouds(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clouds) { m_clouds_vector = clouds; }

void PointPose::setSourceCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in) { m_source = cloud_in; }

void PointPose::setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) { m_cloud_grasp = cloud; }

void PointPose::setRefPlane(pcl::ModelCoefficients::Ptr &plane) { m_plane = plane; }

Eigen::Vector3f PointPose::getTranslation() { return m_trans; }

Eigen::Quaternionf PointPose::getRotation() { return m_rot; }

Eigen::Vector3f PointPose::getDirectionWrinkle()
{
    Eigen::Vector3f value;
    value.x() = m_line.values[3];
    value.y() = m_line.values[4];
    value.z() = m_line.values[5];

    return value;
}

int PointPose::compute(Affine3dVector &transformation_matrix_vector, std::vector<float> &margins)
{
    Eigen::Affine3d matrix;
    float margin_value;
    int counter = m_clouds_vector.size();
    for (int i = 0; i < counter; i++)
    {
        computeGraspPoint(m_clouds_vector[i], matrix, margin_value);
        transformation_matrix_vector.push_back(matrix);
        margins.push_back(margin_value);
    }

    return counter;
}

bool PointPose::computeGraspPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                  Eigen::Affine3d &transformation_matrix, float &margin)
{

    // Create the filtering object
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud);
    proj.setModelCoefficients(m_plane);
    proj.filter(*m_cloud_projected);

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*m_cloud_projected, centroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*m_cloud_projected, centroid, covariance);

    Eigen::Vector3f eigenValues;
    Eigen::Matrix3f eigenVectors;
    pcl::eigen33(covariance, eigenVectors, eigenValues);

    std::vector<int> idx = orderEigenvalues(eigenValues);
    Eigen::Matrix3f rotation;
    rotation.col(0) = eigenVectors.col(idx[0]);
    rotation.col(1) = eigenVectors.col(idx[1]);
    rotation.col(2) = eigenVectors.col(idx[2]);

    m_trans = moveCentroid(centroid, cloud); //move centroid from projected cloud to original cloud_grasp

    //compute point for visualizing coordinate axes
    Eigen::Vector3f directionX, directionZ;
    getCoordinateFrame(m_trans, rotation, directionX, directionZ);
    transformation_matrix = computeTransformation(m_trans, directionX, directionZ);

    // compute margin for the grasp wrt plane
    Eigen::Vector3f n(m_plane->values[0], m_plane->values[1], m_plane->values[2]);
    float p = m_plane->values[3];
    margin = n.dot(m_trans) + p;

    if (margin < 0)
        margin = -margin;

    std::cout << " margin : " << margin << std::endl;

    return true;
}
void PointPose::visualizeGrasp()
{
    pcl::visualization::PCLVisualizer viz("PCL Cloud Result");
    //viz.addCoordinateSystem(0.1);
    viz.setBackgroundColor(1.0f, 1.0f, 1.0f);
    viz.addPointCloud<pcl::PointXYZRGB>(m_source, "source");
    viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0f, 0.7f, 0.0f, "source");
    viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source");

    for (int i = 0; i < m_clouds_vector.size(); i++)
    {
        viz.addPointCloud<pcl::PointXYZ>(m_clouds_vector[i], "cloud_grasp" + std::to_string(i));
        viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_grasp" + std::to_string(i));
        viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0f, 1.0f, 0.0f, "cloud_grasp" + std::to_string(i));

        viz.addSphere(m_cfp_viz[i].o, 0.005, "sphere" + std::to_string(i));
        viz.addArrow(m_cfp_viz[i].x, m_cfp_viz[i].o, 1.0f, 0.0f, 0.0f, false, "x_axis" + std::to_string(i));
        viz.addArrow(m_cfp_viz[i].y, m_cfp_viz[i].o, 0.0f, 1.0f, 0.0f, false, "y_axis" + std::to_string(i));
        viz.addArrow(m_cfp_viz[i].z, m_cfp_viz[i].o, 0.0f, 0.0f, 1.0f, false, "z_axis" + std::to_string(i));
        viz.addText3D("T" + std::to_string(i), m_cfp_viz[i].o, 0.01, 1.0, 0.0, 0., "Text" + std::to_string(i));
    }

    pcl::visualization::PCLVisualizer viz2("PCL Cloud Result 2");
    //viz.addCoordinateSystem(0.1);
    viz2.setBackgroundColor(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < m_clouds_vector.size(); i++)
    {
        viz2.addPointCloud<pcl::PointXYZ>(m_clouds_vector[i], "cloud_grasp" + std::to_string(i));
        viz2.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_grasp" + std::to_string(i));
        viz2.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0f, 1.0f, 0.0f, "cloud_grasp" + std::to_string(i));

        viz2.addSphere(m_cfp_viz[i].o, 0.005, "sphere" + std::to_string(i));
        viz2.addArrow(m_cfp_viz[i].x, m_cfp_viz[i].o, 1.0f, 0.0f, 0.0f, false, "x_axis" + std::to_string(i));
        viz2.addArrow(m_cfp_viz[i].y, m_cfp_viz[i].o, 0.0f, 1.0f, 0.0f, false, "y_axis" + std::to_string(i));
        viz2.addArrow(m_cfp_viz[i].z, m_cfp_viz[i].o, 0.0f, 0.0f, 1.0f, false, "z_axis" + std::to_string(i));
        viz2.addText3D("T" + std::to_string(i), m_cfp_viz[i].o, 0.01, 1.0, 0.0, 0., "Text" + std::to_string(i));
    }
}

void PointPose::visualizeCloudGrasp(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in)
{

    pcl::visualization::PCLVisualizer viz2("PCL Cloud Grasp");
    //viz.addCoordinateSystem(0.1);
    viz2.setBackgroundColor(1.0f, 1.0f, 1.0f);

    viz2.addPointCloud<pcl::PointXYZRGB>(cloud_in, "cloud_in");
    viz2.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0f, 0.7f, 0.0f, "cloud_in");
    viz2.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_in");

    for (int i = 0; i < m_clouds_vector.size(); i++)
    {
        viz2.addPointCloud<pcl::PointXYZ>(m_clouds_vector[i], "cloud_grasp" + std::to_string(i));
        viz2.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_grasp" + std::to_string(i));
        viz2.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0f, 1.0f, 0.0f, "cloud_grasp" + std::to_string(i));

        viz2.addSphere(m_cfp_viz[i].o, 0.005, "sphere" + std::to_string(i));
        viz2.addArrow(m_cfp_viz[i].x, m_cfp_viz[i].o, 1.0f, 0.0f, 0.0f, false, "x_axis" + std::to_string(i));
        viz2.addArrow(m_cfp_viz[i].y, m_cfp_viz[i].o, 0.0f, 1.0f, 0.0f, false, "y_axis" + std::to_string(i));
        viz2.addArrow(m_cfp_viz[i].z, m_cfp_viz[i].o, 0.0f, 0.0f, 1.0f, false, "z_axis" + std::to_string(i));
        //viz2.addText3D("T" + std::to_string(i), m_cfp_viz[i].o, 0.01, 1.0, 0.0, 0., "Text" + std::to_string(i));
    }
}

std::vector<int> PointPose::orderEigenvalues(Eigen::Vector3f eigenValuesPCA)
{
    std::vector<double> v;
    v.push_back(eigenValuesPCA[0]);
    v.push_back(eigenValuesPCA[1]);
    v.push_back(eigenValuesPCA[2]);

    int maxElementIndex = std::max_element(v.begin(), v.end()) - v.begin();
    double maxElement = *std::max_element(v.begin(), v.end());

    int minElementIndex = std::min_element(v.begin(), v.end()) - v.begin();
    double minElement = *std::min_element(v.begin(), v.end());

    int middleElementIndex;
    for (int i = 0; i < 3; i++)
    {
        if (i == maxElementIndex || i == minElementIndex)
            continue;
        middleElementIndex = i;
    }
    v.clear();
    std::vector<int> result;
    result.push_back(maxElementIndex);
    result.push_back(middleElementIndex);
    result.push_back(minElementIndex);

    return result;
}

Eigen::Vector3f PointPose::moveCentroid(Eigen::Vector4f centroid, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(m_cloud_projected);
    pcl::PointXYZ searchPoint;

    searchPoint.x = centroid.x();
    searchPoint.y = centroid.y();
    searchPoint.z = centroid.z();

    int K = 1;
    std::vector<int> indices(K);
    std::vector<float> distances(K);

    if (kdtree.nearestKSearch(searchPoint, K, indices, distances) > 0)
    {
        m_origin = cloud->points[indices[0]];
        _centroid = cloud->points[indices[0]].getVector3fMap();
    }

    return _centroid;
}

void PointPose::getCoordinateFrame(Eigen::Vector3f &centroid, Eigen::Matrix3f &rotation,
                                   Eigen::Vector3f &directionX, Eigen::Vector3f &directionZ)
{
    bool reverse = false;

    pcl::PointXYZ centroidXYZ;
    centroidXYZ.getVector3fMap() = centroid;

    pcl::PointXYZ PointY = pcl::PointXYZ((centroid(0) + rotation.col(0)(0)),
                                         (centroid(1) + rotation.col(0)(1)),
                                         (centroid(2) + rotation.col(0)(2)));

    pcl::PointXYZ PointZ = pcl::PointXYZ((centroid(0) + rotation.col(1)(0)),
                                         (centroid(1) + rotation.col(1)(1)),
                                         (centroid(2) + rotation.col(1)(2)));

    pcl::PointXYZ PointX = pcl::PointXYZ((centroid(0) + rotation.col(2)(0)),
                                         (centroid(1) + rotation.col(2)(1)),
                                         (centroid(2) + rotation.col(2)(2)));

    if (PointX.z < centroid(2))
    {
        PointX = pcl::PointXYZ((centroid(0) - rotation.col(2)(0)),
                               (centroid(1) - rotation.col(2)(1)),
                               (centroid(2) - rotation.col(2)(2)));

        reverse = true;
    }

    CoordinateFramePoints points;
    points.o = centroidXYZ;
    points.x = PointX;
    points.y = PointY;
    points.z = PointZ;

    directionX = points.x.getVector3fMap() - points.o.getVector3fMap();
    directionZ = points.z.getVector3fMap() - points.o.getVector3fMap();

    m_cfp.push_back(points);

    computeCoordinateFramePointsViz(centroid, rotation, reverse);
}

void PointPose::computeCoordinateFramePointsViz(Eigen::Vector3f &centroid, Eigen::Matrix3f &rotation, bool reverse)
{
    float factor = 0.1;
    pcl::PointXYZ centroidXYZ;
    centroidXYZ.getVector3fMap() = centroid;

    pcl::PointXYZ PointY = pcl::PointXYZ((centroid(0) + factor * rotation.col(0)(0)),
                                         (centroid(1) + factor * rotation.col(0)(1)),
                                         (centroid(2) + factor * rotation.col(0)(2)));

    pcl::PointXYZ PointZ = pcl::PointXYZ((centroid(0) + factor * rotation.col(1)(0)),
                                         (centroid(1) + factor * rotation.col(1)(1)),
                                         (centroid(2) + factor * rotation.col(1)(2)));

    pcl::PointXYZ PointX = pcl::PointXYZ((centroid(0) + factor * rotation.col(2)(0)),
                                         (centroid(1) + factor * rotation.col(2)(1)),
                                         (centroid(2) + factor * rotation.col(2)(2)));

    if (reverse)
    {
        PointX = pcl::PointXYZ((centroid(0) - factor * rotation.col(2)(0)),
                               (centroid(1) - factor * rotation.col(2)(1)),
                               (centroid(2) - factor * rotation.col(2)(2)));
    }

    CoordinateFramePoints points;
    points.o = centroidXYZ;
    points.x = PointX;
    points.y = PointY;
    points.z = PointZ;

    m_cfp_viz.push_back(points);
}

Eigen::Affine3d PointPose::computeTransformation(Eigen::Vector3f &centroid, Eigen::Vector3f &directionX, Eigen::Vector3f &directionZ)
{
    Eigen::VectorXd from_line_x, from_line_z, to_line_x, to_line_z;

    from_line_x.resize(6);
    from_line_z.resize(6);
    to_line_x.resize(6);
    to_line_z.resize(6);

    //Origin
    from_line_x << 0, 0, 0, 1, 0, 0;
    from_line_z << 0, 0, 0, 0, 0, 1;

    to_line_x.head<3>() = centroid.cast<double>();
    to_line_x.tail<3>() = directionX.cast<double>();

    to_line_z.head<3>() = centroid.cast<double>();
    to_line_z.tail<3>() = directionZ.cast<double>();

    Eigen::Affine3d transformation;
    if (!pcl::transformBetween2CoordinateSystems(from_line_x, from_line_z, to_line_x, to_line_z, transformation))
    {
        PCL_WARN("Transformation not found!\n");
    }
    return transformation;
}