#include "../include/segmentation.h"

// INPUT DATA
void SegFilter::setSourceCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) { source_ = cloud; }

void SegFilter::setLine1(Eigen::Vector3f &pt, Eigen::Vector3f &dir) { line1_.values = {pt.x(), pt.y(), pt.z(), dir.x(), dir.y(), dir.z()}; }

void SegFilter::setLine2(Eigen::Vector3f &pt, Eigen::Vector3f &dir) { line2_.values = {pt.x(), pt.y(), pt.z(), dir.x(), dir.y(), dir.z()}; }

void SegFilter::setDrumDimensions(float radius, float depth)
{
    drumDepth_ = depth;
    radiusDrum_ = radius;
};

void SegFilter::setDistanceDrumCenter(float distanceCenter) { distanceCenterDrum_ = distanceCenter; }

// COMPUTE FUNCTION
void SegFilter::compute(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{

    //computeDrumAxes(line1_, line2_);

    Eigen::Vector3f axis_dir = {line1_.values[3], line1_.values[4], line1_.values[5]};
    Eigen::Vector3f origin = {line1_.values[0], line1_.values[1], line1_.values[2]};

    Eigen::Vector3f center = origin + axis_dir * distanceCenterDrum_;
    centerPoint_.getVector3fMap() = center;

    segPoint1_.getVector3fMap() = center - (drumDepth_ / 2) * axis_dir;
    segPoint2_.getVector3fMap() = center + (drumDepth_ / 2) * axis_dir;

    Eigen::Vector3f vector_dir = {line2_.values[3], line2_.values[4], line2_.values[5]};

    pointsHull1_ = calculateHullPoints(segPoint1_, axis_dir, vector_dir, radiusDrum_);
    pointsHull2_ = calculateHullPoints(segPoint2_, axis_dir, vector_dir, radiusDrum_);

    combineHullPoints(pointsHull1_, pointsHull2_, hull_vertices_);

    convexHullCrop(source_, hull_vertices_, cloud_output_);

    cloud = cloud_output_;
}

// VISUALIZATION
void SegFilter::visualizeSeg(bool flagSpin)
{
    pcl::visualization::PCLVisualizer vizSource("PCL Transformation");
    //vizSource.addCoordinateSystem(0.1, "coord");
    vizSource.setBackgroundColor(1.0f, 1.0f, 1.0f);
    vizSource.addPointCloud<pcl::PointXYZRGB>(source_, "source_");
    vizSource.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0f, 0.7f, 0.0f, "source_");

    pcl::PointXYZ line1_0 = pcl::PointXYZ(line1_.values[0], line1_.values[1], line1_.values[2]);
    pcl::PointXYZ line2_0 = pcl::PointXYZ(line2_.values[0], line2_.values[1], line2_.values[2]);

    pcl::PointXYZ line1_1 = pcl::PointXYZ(line1_0.x + 0.75 * line1_.values[3],
                                          line1_0.y + 0.75 * line1_.values[4],
                                          line1_0.z + 0.75 * line1_.values[5]);
    pcl::PointXYZ line2_1 = pcl::PointXYZ(line2_0.x + 0.25 * line2_.values[3],
                                          line2_0.y + 0.25 * line2_.values[4],
                                          line2_0.z + 0.25 * line2_.values[5]);

    vizSource.addLine(line1_0, line1_1, 1.0f, 0.0f, 0.0f, "line1");
    vizSource.addLine(line2_0, line2_1, 0.0f, 0.0f, 1.0f, "line2");

    vizSource.addPointCloud<pcl::PointXYZRGB>(hull_vertices_, "hull");
    vizSource.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0f, 0.0f, 0.0f, "hull");
    vizSource.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "hull");

    pcl::visualization::PCLVisualizer vizHull("PCL Result Hull");
    //vizSource.addCoordinateSystem(0.1, "coord");
    vizHull.setBackgroundColor(1.0f, 1.0f, 1.0f);
    vizHull.addPointCloud<pcl::PointXYZRGB>(cloud_output_, "cloud_output_");
    vizHull.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0f, 0.0f, 0.0f, "cloud_output_");

    if (flagSpin)
        vizSource.spin();
}

// PRIVATE FUNCTIONS:
std::vector<pcl::PointXYZ> SegFilter::calculateHullPoints(pcl::PointXYZ &point1, Eigen::Vector3f &axis,
                                                          Eigen::Vector3f &vector, float radius_cylinder)
{

    //extra padding
    radius_cylinder += 0.05; // <<<----------------------------------

    std::vector<pcl::PointXYZ> newPoints;
    pcl::PointXYZ new_point;

    // Rodrigues' rotation formula

    // angle to rotate
    float theta = M_PI / 10;

    // unit versor k
    Eigen::Vector3f k = axis;
    k.normalize();

    // vector to rotate V
    Eigen::Vector3f V = vector;
    Eigen::Vector3f V_rot;

    V_rot = V * cos(-M_PI_2) + (k.cross(V)) * sin(-M_PI_2) + k * (k.dot(V)) * (1 - cos(-M_PI_2));
    V_rot.normalize();
    new_point.x = point1.x + radius_cylinder * V_rot.x();
    new_point.y = point1.y + radius_cylinder * V_rot.y();
    new_point.z = point1.z + radius_cylinder * V_rot.z();

    newPoints.push_back(new_point);
    V = V_rot;

    for (int c = 0; c < 10; c++)
    {

        V_rot = V * cos(theta) + (k.cross(V)) * sin(theta) + k * (k.dot(V)) * (1 - cos(theta));
        V_rot.normalize();

        new_point.x = point1.x + radius_cylinder * V_rot.x();
        new_point.y = point1.y + radius_cylinder * V_rot.y();
        new_point.z = point1.z + radius_cylinder * V_rot.z();

        newPoints.push_back(new_point);
        V = V_rot;
    }

    newPoints.push_back(point1);

    return newPoints;
}

void SegFilter::combineHullPoints(std::vector<pcl::PointXYZ> &p1, std::vector<pcl::PointXYZ> &p2,
                                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    cloud->width = p1.size() + p2.size();
    cloud->height = 1;
    cloud->resize(cloud->width * cloud->height);

    for (int i = 0; i < p1.size(); i++)
    {
        cloud->points[i].x = p1[i].x;
        cloud->points[i].y = p1[i].y;
        cloud->points[i].z = p1[i].z;
    }

    for (int i = 0; i < p2.size(); i++)
    {
        cloud->points[i + p1.size()].x = p2[i].x;
        cloud->points[i + p1.size()].y = p2[i].y;
        cloud->points[i + p1.size()].z = p2[i].z;
    }
}

void SegFilter::convexHullCrop(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                               pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_vertices,
                               pcl::PointCloud<pcl::PointXYZRGB>::Ptr &hull_result)
{
    pcl::CropHull<pcl::PointXYZRGB> cropHullFilter;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_hull(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::vector<pcl::Vertices> hullPolygons;

    // setup hull filter
    pcl::ConvexHull<pcl::PointXYZRGB> cHull;
    cHull.setInputCloud(cloud_vertices);
    cHull.reconstruct(*points_hull, hullPolygons);

    cropHullFilter.setHullIndices(hullPolygons);
    cropHullFilter.setHullCloud(points_hull);
    cropHullFilter.setCropOutside(true);

    //filter points
    cropHullFilter.setInputCloud(cloud);
    cropHullFilter.filter(*hull_result);
}

//---------------------------------------------------------------------------------------------
//functions to compute drum axes, only for debug
/*
void SegFilter::computeDrumAxes(pcl::ModelCoefficients &line1, pcl::ModelCoefficients &line2)
{

    //add fake points along z axis

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_fake(new pcl::PointCloud<pcl::PointXYZ>);
    int numPoints_1 = 50, numPoints_2 = 25;

    cloud_fake->width = numPoints_1 + numPoints_2;
    cloud_fake->height = 1;
    cloud_fake->resize(cloud_fake->height * cloud_fake->width);

    for (int k = 0; k < cloud_fake->size(); k++)
    {
        if (k <= numPoints_1)
        {
            cloud_fake->points[k].x = 0;
            cloud_fake->points[k].y = 0;
            cloud_fake->points[k].z = 0 + k / 100.0f;
        }
        else
        {
            cloud_fake->points[k].x = 0;
            cloud_fake->points[k].y = 0 + (k - numPoints_2) / 100.0f;
            cloud_fake->points[k].z = 0;
        }

        //std::cout << cloud_fake->points[k].getVector3fMap() << std::endl;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_fake_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    transformation(cloud_fake, cloud_fake_transformed);

    // line detection
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::ExtractIndices<pcl::PointXYZ> extract;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);

    seg.setInputCloud(cloud_fake_transformed);
    seg.segment(*inliers, line1);

    extract.setInputCloud(cloud_fake_transformed);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_fake_transformed);

    seg.setInputCloud(cloud_fake_transformed);
    seg.segment(*inliers, line2);

    // origin
    line1.values[0] = 0;
    line1.values[1] = 0.09;
    line1.values[2] = 0.11;
    line2.values[0] = 0;
    line2.values[1] = 0.09;
    line2.values[2] = 0.11;

    for (int i = 0; i < line1.values.size(); i++)
    {
        std::cout << line1.values[i] << std::endl;
    }
    for (int i = 0; i < line2.values.size(); i++)
    {
        std::cout << line2.values[i] << std::endl;
    }
}

void SegFilter::transformation(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
{
    float theta = 0.9075; // 0.9075 --- 52 deg

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << 0.0, 0.0, 0.0;
    transform.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitX()));
    pcl::transformPointCloud(*cloud, *cloud_out, transform);
}
*/