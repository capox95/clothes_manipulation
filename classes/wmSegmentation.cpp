#include "../include/wmSegmentation.h"
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/common/intersections.h>

void Processing::setSceneCloud(PointCloudT::Ptr scene) { scene_ = scene; }

void Processing::setObjectCloud(PointCloudT::Ptr object) { object_result_ = object; }

pcl::ModelCoefficients::Ptr Processing::getPlaneUsed() { return plane_; }

PointCloudIntT::Ptr Processing::getSceneMap() { return scene_map_; }

void Processing::getPointPlaneDistanceCloud(PointCloudT::Ptr &obj, PointCloudT::Ptr &scene, PointCloudT::Ptr scene_proj,
                                            PointCloudIntT::Ptr &cloud_out, pcl::ModelCoefficients::Ptr &plane)
{
    float threshold = 0.05;

    Eigen::Vector3f x0, y0;
    Eigen::Vector3f n(plane->values[0], plane->values[1], plane->values[2]);
    float p = plane->values[3];
    float distance, distance_obj;

    cloud_out->width = scene->width;
    cloud_out->height = scene->height;
    cloud_out->resize(scene->width * scene->height);

    pcl::search::KdTree<PointNT> kdtree;
    PointNT searchPoint;
    kdtree.setInputCloud(scene_proj);

    int K = 1;
    std::vector<int> indices(K);
    std::vector<float> distances(K);

    for (int i = 0; i < obj->size(); i++)
    {
        searchPoint = obj->points[i];

        if (kdtree.nearestKSearch(searchPoint, K, indices, distances) > 0)
        {
            if ((sqrt(distances[0]) < threshold) &&
                (!std::isnan(scene->points[indices[0]].x)) &&
                (!std::isnan(scene->points[indices[0]].y)) && (!std::isnan(scene->points[indices[0]].z)))
            {
                // computation for scene cloud
                x0 = scene->points[indices[0]].getVector3fMap();
                distance = n.dot(x0) + p;

                // computation for obj cloud
                y0 = obj->points[i].getVector3fMap();
                distance_obj = n.dot(y0) + p;

                // new cloud
                cloud_out->points[i].x = scene->points[indices[0]].x;
                cloud_out->points[i].y = scene->points[indices[0]].y;
                cloud_out->points[i].z = scene->points[indices[0]].z;

                if (distance_obj - distance > 0)
                    cloud_out->points[i].intensity = distance_obj - distance;
                else
                    cloud_out->points[i].intensity = 0;
            }
        }
        indices.clear();
        distances.clear();
    }
}

void Processing::computeDiffProjected(PointCloudT::Ptr obj, PointCloudT::Ptr scene,
                                      PointCloudIntT::Ptr cloud_output, PointCloudT::Ptr hull, pcl::ModelCoefficients::Ptr &plane)
{
    PointCloudT::Ptr scene_proj(new PointCloudT);
    PointCloudT::Ptr obj_proj(new PointCloudT);

    pcl::SACSegmentation<PointNT> seg;
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(200);
    seg.setDistanceThreshold(0.05);
    seg.setInputCloud(obj);
    // Obtain the plane inliers and coefficients
    seg.segment(*inliers_plane, *coefficients_plane);
    //std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

    // Create the filtering object
    pcl::ProjectInliers<PointNT> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(scene);
    proj.setModelCoefficients(coefficients_plane);
    proj.filter(*scene_proj);

    proj.setInputCloud(obj);
    proj.filter(*obj_proj);

    // Create a Concave Hull representation of the projected inliers
    pcl::ConcaveHull<PointNT> chull;
    chull.setInputCloud(obj_proj);
    chull.setAlpha(0.1);
    chull.reconstruct(*hull);

    getPointPlaneDistanceCloud(obj, scene, scene_proj, cloud_output, coefficients_plane);

    plane = coefficients_plane;
}

void Processing::lineDetection(PointCloudT::Ptr &hull, std::vector<pcl::ModelCoefficients> &lines, pcl::PointXYZ &p1, pcl::PointXYZ &p2)
{
    pcl::ModelCoefficients line;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<PointNT> seg;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    for (int i = 0; i < 3; i++)
    {
        seg.setInputCloud(hull);
        seg.segment(*inliers, line);

        if (inliers->indices.size() == 0)
        {
            std::cerr << "Could not estimate a linear model for the given dataset." << std::endl;
        }
        pcl::ExtractIndices<PointNT> extract;
        // Extract the inliers
        extract.setInputCloud(hull);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*hull);

        lines.push_back(line);
    }

    if (lines.size() != 3)
        PCL_WARN("number of lines is not 3!\n");

    Eigen::Vector3f line0_dir = {lines[0].values[3], lines[0].values[4], lines[0].values[5]};
    Eigen::Vector3f line1_dir = {lines[1].values[3], lines[1].values[4], lines[1].values[5]};
    Eigen::Vector3f line2_dir = {lines[2].values[3], lines[2].values[4], lines[2].values[5]};

    double angle01 = acos(line0_dir.dot(line1_dir));
    double angle12 = acos(line1_dir.dot(line2_dir));
    double angle02 = acos(line0_dir.dot(line2_dir));

    std::cout << angle01 << "; " << angle12 << "; " << angle02 << std::endl;

    std::vector<Eigen::Vector4f> points;
    float eps = 0.03;
    Eigen::Vector4f ptemp;

    if (fabs(angle01 - M_PI_2) < 0.5)
    {
        pcl::lineWithLineIntersection(lines[0], lines[1], ptemp, eps);
        points.push_back(ptemp);
    }
    if (fabs(angle02 - M_PI_2) < 0.5)
    {
        pcl::lineWithLineIntersection(lines[0], lines[2], ptemp, eps);
        points.push_back(ptemp);
    }
    if (fabs(angle12 - M_PI_2) < 0.5)
    {
        pcl::lineWithLineIntersection(lines[1], lines[2], ptemp, eps);
        points.push_back(ptemp);
    }

    if (points.size() > 2)
        PCL_WARN("obtained different than 2 intersection points\n");

    p1.getVector4fMap() = points[0];
    p2.getVector4fMap() = points[1];
}

void Processing::segmentSceneCloud(PointCloudIntT::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_seg)
{
    int counter = 0;
    for (PointT point : cloud->points)
        if (point.intensity > 0.01)
            counter++;

    //std::cout << counter << std::endl;

    cloud_seg->width = counter;
    cloud_seg->height = 1;
    cloud_seg->resize(cloud_seg->width);

    pcl::PointXYZRGB pointTmp;
    for (int i = 0; i < cloud->size(); i++)
    {
        if (cloud->points[i].intensity > 0.01)
        {
            pointTmp.x = cloud->points[i].x;
            pointTmp.y = cloud->points[i].y;
            pointTmp.z = cloud->points[i].z;
            cloud_seg->points.push_back(pointTmp);
        }
    }
}

void Processing::updateVerticesCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr vertices, pcl::PointCloud<pcl::PointXYZ>::Ptr update,
                                     pcl::ModelCoefficients::Ptr &plane)
{
    Eigen::Vector3f n = {plane->values[0], plane->values[1], plane->values[2]};
    n.normalize();
    pcl::PointXYZ tmp_point;
    for (int k = 0; k < vertices->size(); k++)
    {
        tmp_point.getVector3fMap() = vertices->points[k].getVector3fMap() - 0.2 * n;
        update->points.push_back(tmp_point);
        tmp_point.getVector3fMap() = vertices->points[k].getVector3fMap() + 0.2 * n;
        update->points.push_back(tmp_point);
    }
}

void Processing::croppingWithHull(pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_updated,
                                  PointCloudIntT::Ptr cloud,
                                  PointCloudIntT::Ptr hull)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_vertices(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*vertices_updated, *cloud_vertices);

    pcl::CropHull<pcl::PointXYZI> cropHullFilter;
    pcl::PointCloud<pcl::PointXYZI>::Ptr points_hull(new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<pcl::Vertices> hullPolygons;

    // setup hull filter
    pcl::ConvexHull<pcl::PointXYZI> cHull;
    cHull.setInputCloud(cloud_vertices);
    cHull.reconstruct(*points_hull, hullPolygons);

    cropHullFilter.setHullIndices(hullPolygons);
    cropHullFilter.setHullCloud(points_hull);
    cropHullFilter.setCropOutside(true);

    //filter points
    cropHullFilter.setInputCloud(cloud);
    cropHullFilter.filter(*hull);

    std::cout << std::endl;
    std::cout << "hull result points: " << hull->size() << std::endl;
}

void Processing::combineHulls(PointCloudIntT::Ptr hull1, PointCloudIntT::Ptr hull2, PointCloudIntT::Ptr result)
{
    pcl::copyPointCloud(*hull1, *result);

    int hull1_size = hull1->size();

    result->height = 1;
    result->width = hull1->width + hull2->width;
    result->resize(result->height * result->width);

    for (int i = 0; i < hull2->size(); i++)
        result->points[hull1_size + i] = hull2->points[i];
}

PointCloudIntT::Ptr Processing::convexCrop(PointCloudT::Ptr obj, PointCloudIntT::Ptr cloud,
                                           pcl::PointXYZ &p1, pcl::PointXYZ &p2, pcl::ModelCoefficients::Ptr &plane)
{
    std::cout << "cloud points: " << cloud->points.size() << std::endl;

    pcl::compute3DCentroid(*object_result_, centroid_);
    P11_.getVector4fMap() = p1.getVector4fMap() + 2 * (centroid_ - p1.getVector4fMap());
    P22_.getVector4fMap() = p2.getVector4fMap() + 2 * (centroid_ - p2.getVector4fMap());

    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices1(new pcl::PointCloud<pcl::PointXYZ>);
    vertices1->points.push_back(p1);
    vertices1->points.push_back(p2);
    vertices1->points.push_back(P11_);
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices1_updated(new pcl::PointCloud<pcl::PointXYZ>);
    updateVerticesCloud(vertices1, vertices1_updated, plane);
    PointCloudIntT::Ptr hull1(new PointCloudIntT);
    croppingWithHull(vertices1_updated, cloud, hull1);

    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices2(new pcl::PointCloud<pcl::PointXYZ>);
    vertices2->points.push_back(p1);
    vertices2->points.push_back(p2);
    vertices2->points.push_back(P22_);
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices2_updated(new pcl::PointCloud<pcl::PointXYZ>);
    updateVerticesCloud(vertices2, vertices2_updated, plane);
    PointCloudIntT::Ptr hull2(new PointCloudIntT);
    croppingWithHull(vertices2_updated, cloud, hull2);

    PointCloudIntT::Ptr result_hull(new PointCloudIntT);
    combineHulls(hull1, hull2, result_hull);

    return result_hull;
}

void Processing::compute(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_seg)
{
    computeDiffProjected(object_result_, scene_, scene_map_, cloud_hull_, plane_);

    lineDetection(cloud_hull_, lines_, P1_, P2_);

    combined_hulls_ = convexCrop(object_result_, scene_map_, P1_, P2_, plane_);

    segmentSceneCloud(combined_hulls_, cloud_seg);
}

void Processing::visualize()
{
    //
    pcl::visualization::PCLVisualizer viz("Visualizer");
    //viz.addCoordinateSystem(0.1, "coordinate");
    viz.setBackgroundColor(1.0, 1.0, 1.0);

    viz.addPointCloud<PointNT>(scene_, "scene");
    viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0f, 0.7f, 0.0f, "scene");
    viz.addPointCloud<PointNT>(object_result_, "object_result");
    viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.7f, 0.0f, 0.0f, "object_result");

    pcl::visualization::PCLVisualizer vizScene("Scene Map");
    vizScene.setBackgroundColor(1.0, 1.0, 1.0);
    //vizScene.addPointCloud<PointT>(diff_scene, "diff_scene");
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distribution(scene_map_, "intensity");
    vizScene.addPointCloud<pcl::PointXYZI>(scene_map_, intensity_distribution, "diff_scene");

    /*

    */
    /*
    pcl::visualization::PCLVisualizer vizProj("vizHull");
    vizProj.setBackgroundColor(1.0, 1.0, 1.0);
    vizProj.addPointCloud<PointNT>(object_result_, "object_result_");
    vizProj.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.7f, 0.0f, 0.0f, "object_result_");
    vizProj.addPointCloud<PointNT>(cloud_hull_tmp, "cloud_hull_");
    vizProj.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0f, 1.0f, 1.0f, "cloud_hull_");
    vizProj.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "cloud_hull_");

    vizProj.addSphere(P1_, 0.01, 0.0, 1.0, 0.0, "p1");
    vizProj.addText3D("P1", P1_, 0.02, 0.0, 0.7, 0.0, "p1_text");
    vizProj.addSphere(P2_, 0.01, 0.0, 1.0, 0.0, "p2");
    vizProj.addText3D("P2", P2_, 0.02, 0.0, 0.7, 0.0, "p2_text");

    vizProj.addSphere(P11_, 0.01, 0.0, 1.0, 0.0, "p11");
    vizProj.addText3D("P1'", P11_, 0.02, 0.0, 0.7, 0.0, "p11_text");

    vizProj.addSphere(P22_, 0.01, 0.0, 1.0, 0.0, "p22");
    vizProj.addText3D("P2'", P22_, 0.02, 0.0, 0.7, 0.0, "p22_text");

    pcl::PointXYZ centroidPoint;
    centroidPoint.getVector4fMap() = centroid_;
    vizProj.addSphere(centroidPoint, 0.01, 0.0, 1.0, 0.0, "centroid");
    vizProj.addText3D("C", centroidPoint, 0.02, 0.0, 0.7, 0.0, "c_text");
    */
    //vizScene.addLine(lines_[0], "line0", 0);
    //vizScene.addLine(lines_[1], "line1", 0);
    //vizScene.addLine(lines_[2], "line2", 0);

    pcl::visualization::PCLVisualizer vizHull("vizHull");
    vizHull.setBackgroundColor(1.0, 1.0, 1.0);
    //vizHull.addPointCloud<PointT>(combined_hulls_, "combined_hulls_");
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionC(combined_hulls_, "intensity");
    vizHull.addPointCloud<pcl::PointXYZI>(combined_hulls_, intensity_distributionC, "combined_hulls_");
}
