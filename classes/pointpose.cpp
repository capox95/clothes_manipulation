#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/pca.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/visualization/cloud_viewer.h>

#include "../include/pointpose.h"

void PointPoseDrum::setSourceCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in) { m_source = cloud_in; }

void PointPoseDrum::setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) { m_cloud_grasp = cloud; }

void PointPoseDrum::setDrumAxis(pcl::ModelCoefficients &axis) { m_axis = axis; }

void PointPoseDrum::setInputVectorClouds(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clouds) { m_clouds_vector = clouds; }

int PointPoseDrum::compute(Vector3fVector &pointsOnAxis, Affine3dVector &transformation_matrix_vector)
{
    Eigen::Affine3d matrix;
    Eigen::Vector3f point;
    int counter = m_clouds_vector.size();
    for (int i = 0; i < counter; i++)
    {
        computeGraspPoint(m_clouds_vector[i], point, matrix);
        pointsOnAxis.push_back(point);
        transformation_matrix_vector.push_back(matrix);
    }

    return counter;
}

bool PointPoseDrum::computeGraspPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                      Eigen::Vector3f &point, Eigen::Affine3d &transformation_matrix)
{

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    //compute ref plane for coordinate system
    pcl::ModelCoefficients plane;
    pcl::PointXYZ pointOnTheLine;
    computeRefPlane(m_axis, centroid, plane, pointOnTheLine);
    m_pointOnAxis.push_back(pointOnTheLine);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);
    projectPointsOntoPlane(cloud, plane, cloud_projected);

    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud_projected, centroid, covariance);

    Eigen::Vector3f eigenValues;
    Eigen::Matrix3f eigenVectors;
    pcl::eigen33(covariance, eigenVectors, eigenValues);

    std::vector<int> idx = orderEigenvalues(eigenValues);
    Eigen::Matrix3f rotation;
    rotation.col(0) = eigenVectors.col(idx[0]);
    rotation.col(1) = eigenVectors.col(idx[1]);
    rotation.col(2) = eigenVectors.col(idx[2]);

    m_trans = centroid.head<3>();

    Eigen::Vector3f directionX, directionZ;
    getCoordinateFrame(m_trans, rotation, pointOnTheLine, directionX, directionZ);

    transformation_matrix = computeTransformation(m_trans, directionX, directionZ);
    point = pointOnTheLine.getVector3fMap();

    return true;
}

void PointPoseDrum::visualizeGrasp()
{
    pcl::visualization::PCLVisualizer viz("PCL Cloud Result");
    //viz.addCoordinateSystem(0.1);
    viz.setBackgroundColor(1.0f, 1.0f, 1.0f);
    viz.addPointCloud<pcl::PointXYZRGB>(m_source, "source");
    viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0f, 0.7f, 0.0f, "source");

    for (int i = 0; i < m_clouds_vector.size(); i++)
    {
        viz.addPointCloud<pcl::PointXYZ>(m_clouds_vector[i], "cloud_grasp" + std::to_string(i));
        viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_grasp" + std::to_string(i));
        viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0f, 1.0f, 0.0f, "cloud_grasp" + std::to_string(i));

        viz.addSphere(m_cfp_viz[i].o, 0.005, "sphere" + std::to_string(i));
        viz.addArrow(m_cfp_viz[i].x, m_cfp_viz[i].o, 1.0f, 0.0f, 0.0f, false, "x_axis" + std::to_string(i));
        viz.addArrow(m_cfp_viz[i].y, m_cfp_viz[i].o, 0.0f, 1.0f, 0.0f, false, "y_axis" + std::to_string(i));
        viz.addArrow(m_cfp_viz[i].z, m_cfp_viz[i].o, 0.0f, 0.0f, 1.0f, false, "z_axis" + std::to_string(i));

        viz.addSphere(m_pointOnAxis[i], 0.01, 1.0f, 0.0f, 0.0f, "pointOnAxis" + std::to_string(i));
    }

    pcl::PointXYZ line1_0 = pcl::PointXYZ(m_axis.values[0], m_axis.values[1], m_axis.values[2]);

    pcl::PointXYZ line1_1 = pcl::PointXYZ(line1_0.x + 0.75 * m_axis.values[3],
                                          line1_0.y + 0.75 * m_axis.values[4],
                                          line1_0.z + 0.75 * m_axis.values[5]);

    viz.addLine(line1_0, line1_1, 1.0f, 0.0f, 0.0f, "line1");

}

std::vector<int> PointPoseDrum::orderEigenvalues(Eigen::Vector3f eigenValuesPCA)
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

void PointPoseDrum::getCoordinateFrame(Eigen::Vector3f &centroid, Eigen::Matrix3f &rotation, pcl::PointXYZ &pointOnTheLine,
                                       Eigen::Vector3f &directionX, Eigen::Vector3f &directionZ)
{
    bool reverse = false;

    pcl::PointXYZ centroidXYZ;
    centroidXYZ.getVector3fMap() = centroid;
    float normPO = (centroidXYZ.getVector3fMap() - pointOnTheLine.getVector3fMap()).norm();

    pcl::PointXYZ PointX = pcl::PointXYZ((centroid(0) + rotation.col(0)(0)),
                                         (centroid(1) + rotation.col(0)(1)),
                                         (centroid(2) + rotation.col(0)(2)));

    float normPX = (PointX.getVector3fMap() - pointOnTheLine.getVector3fMap()).norm();
    if ((normPX - normPO) < 0.9)
    {
        PointX = pcl::PointXYZ((centroid(0) - rotation.col(0)(0)),
                               (centroid(1) - rotation.col(0)(1)),
                               (centroid(2) - rotation.col(0)(2)));
        reverse = true;
    }

    pcl::PointXYZ PointZ = pcl::PointXYZ((centroid(0) + rotation.col(1)(0)),
                                         (centroid(1) + rotation.col(1)(1)),
                                         (centroid(2) + rotation.col(1)(2)));

    pcl::PointXYZ PointY = pcl::PointXYZ((centroid(0) + rotation.col(2)(0)),
                                         (centroid(1) + rotation.col(2)(1)),
                                         (centroid(2) + rotation.col(2)(2)));

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

void PointPoseDrum::computeCoordinateFramePointsViz(Eigen::Vector3f &centroid, Eigen::Matrix3f &rotation, bool reverse)
{
    float factor = 0.1;
    pcl::PointXYZ centroidXYZ;
    centroidXYZ.getVector3fMap() = centroid;

    pcl::PointXYZ PointX = pcl::PointXYZ((centroid(0) + factor * rotation.col(0)(0)),
                                         (centroid(1) + factor * rotation.col(0)(1)),
                                         (centroid(2) + factor * rotation.col(0)(2)));

    if (reverse)
    {
        PointX = pcl::PointXYZ((centroid(0) - factor * rotation.col(0)(0)),
                               (centroid(1) - factor * rotation.col(0)(1)),
                               (centroid(2) - factor * rotation.col(0)(2)));
    }

    pcl::PointXYZ PointZ = pcl::PointXYZ((centroid(0) + factor * rotation.col(1)(0)),
                                         (centroid(1) + factor * rotation.col(1)(1)),
                                         (centroid(2) + factor * rotation.col(1)(2)));

    pcl::PointXYZ PointY = pcl::PointXYZ((centroid(0) + factor * rotation.col(2)(0)),
                                         (centroid(1) + factor * rotation.col(2)(1)),
                                         (centroid(2) + factor * rotation.col(2)(2)));

    CoordinateFramePoints points;
    points.o = centroidXYZ;
    points.x = PointX;
    points.y = PointY;
    points.z = PointZ;

    m_cfp_viz.push_back(points);
}

Eigen::Affine3d PointPoseDrum::computeTransformation(Eigen::Vector3f &centroid, Eigen::Vector3f &directionX, Eigen::Vector3f &directionZ)
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

void PointPoseDrum::computeRefPlane(pcl::ModelCoefficients &axis, Eigen::Vector4f &centroid,
                                    pcl::ModelCoefficients &plane, pcl::PointXYZ &pointOnTheLine)
{
    Eigen::Vector3f line_pt, line_dir;

    // format the line considered into vector4f
    line_pt[0] = axis.values[0];
    line_pt[1] = axis.values[1];
    line_pt[2] = axis.values[2];

    line_dir[0] = axis.values[3];
    line_dir[1] = axis.values[4];
    line_dir[2] = axis.values[5];

    //project points onto the line
    Eigen::Vector3f LP;
    LP = centroid.head<3>() - line_pt;

    pointOnTheLine.getVector3fMap() = line_pt + LP.dot(line_dir) * line_dir;

    Eigen::Vector3f normal = centroid.head<3>() - pointOnTheLine.getVector3fMap();
    float d = centroid.head<3>().norm();
    plane.values = {normal.x(), normal.y(), normal.z(), -0.1};
}

void PointPoseDrum::projectPointsOntoPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::ModelCoefficients &plane,
                                           pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_projected)
{
    pcl::ModelCoefficients::Ptr plane_ptr(new pcl::ModelCoefficients);
    plane_ptr->values = plane.values;

    // Create the filtering object
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud);
    proj.setModelCoefficients(plane_ptr);
    proj.filter(*cloud_projected);
}