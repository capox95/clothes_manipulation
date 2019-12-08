#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/pca.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/distances.h>
#include <pcl/common/centroid.h>
#include <pcl/common/geometry.h>

#include "../include/entropy.h"

void EntropyFilterDrum::setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in) { m_source = cloud_in; }

void EntropyFilterDrum::setDrumAxis(pcl::ModelCoefficients line) { m_line = line; }

void EntropyFilterDrum::setEntropyThreshold(float entropy_th) { m_entropy_threshold = entropy_th; }

void EntropyFilterDrum::setKLocalSearch(int K) { m_KNN = K; }

void EntropyFilterDrum::setCurvatureThreshold(float curvature_th) { m_curvature_threshold = curvature_th; }

void EntropyFilterDrum::setDepthThreshold(float depth_th) { m_depth_threshold = depth_th; };

void EntropyFilterDrum::setDrumRadius(float radius) { m_drum_radius = radius; };

bool EntropyFilterDrum::compute(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clouds_out)
{
    //downsample(m_source, m_leafsize, m_cloud_downsample);

    computePolyFitting(m_source, m_mls_points);
    divideCloudNormals(m_mls_points, m_mls_cloud, m_mls_normals);
    getSpherical(m_mls_normals, m_spherical);

    //Depth

    // update depth interval
    m_depth_interval = m_drum_radius - m_depth_threshold;
    computeDepthMap(m_mls_cloud, m_cloud_depth, m_line);

    //Combine
    combineDepthAndCurvatureInfo(m_cloud_depth, m_mls_normals, m_cloud_combined);

    local_search(m_mls_cloud, m_spherical, m_cloud_combined);
    normalizeEntropy(m_spherical);

    if (_max_entropy < 1.0)
    {
        PCL_WARN("Entropy too small!\n");
        return false;
    }

    segmentCloudEntropy(m_mls_points, m_spherical, m_cloud_seg, m_entropy_threshold);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds_connected;
    alternativeConnectedComponets(m_cloud_seg, clouds_connected);

    if (_flag_optimization)
        optimizeNumberOfClouds(clouds_connected, clouds_out);
    else
        clouds_out = clouds_connected;

    return true;
}

void EntropyFilterDrum::visualizeAll(bool flag)
{

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_curvature(new pcl::PointCloud<pcl::PointXYZI>);
    colorMapCurvature(cloud_curvature);
    pcl::visualization::PCLVisualizer vizC("PCL Curvature Map");
    vizC.setBackgroundColor(1.0f, 1.0f, 1.0f);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionCurvature(cloud_curvature, "intensity");
    vizC.addPointCloud<pcl::PointXYZI>(cloud_curvature, intensity_distributionCurvature, "cloud_mapCurvature");
    vizC.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_mapCurvature");

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_entropy(new pcl::PointCloud<pcl::PointXYZI>);
    colorMapEntropy(cloud_entropy);
    pcl::visualization::PCLVisualizer vizE("PCL Entropy Map");
    vizE.setBackgroundColor(1.0f, 1.0f, 1.0f);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionEntropy(cloud_entropy, "intensity");
    vizE.addPointCloud<pcl::PointXYZI>(cloud_entropy, intensity_distributionEntropy, "cloud_mapEntropy");
    vizE.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_mapEntropy");

    if (flag)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_inclination(new pcl::PointCloud<pcl::PointXYZI>);
        colorMapInclination(cloud_inclination);
        pcl::visualization::PCLVisualizer vizI("PCL Inclination Map");
        vizI.setBackgroundColor(1.0f, 1.0f, 1.0f);
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionInclination(cloud_inclination, "intensity");
        vizI.addPointCloud<pcl::PointXYZI>(cloud_inclination, intensity_distributionInclination, "sample cloud_mapInclination");

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_azimuth(new pcl::PointCloud<pcl::PointXYZI>);
        colorMapAzimuth(cloud_azimuth);
        pcl::visualization::PCLVisualizer vizA("PCL Azimuth Map");
        vizA.setBackgroundColor(1.0f, 1.0f, 1.0f);
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionAzimuth(cloud_azimuth, "intensity");
        vizA.addPointCloud<pcl::PointXYZI>(cloud_azimuth, intensity_distributionAzimuth, "sample cloud_mapAzimuth");
    }

    pcl::visualization::PCLVisualizer vizDepth("PCL Depth Map");
    vizDepth.setBackgroundColor(1.0f, 1.0f, 1.0f);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionDepth(m_cloud_depth, "intensity");
    vizDepth.addPointCloud<pcl::PointXYZI>(m_cloud_depth, intensity_distributionDepth, "cloud_depth");
    vizDepth.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_depth");

    pcl::visualization::PCLVisualizer vizCombined("PCL Combined Map");
    vizCombined.setBackgroundColor(1.0f, 1.0f, 1.0f);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distributionCombinede(m_cloud_combined, "intensity");
    vizCombined.addPointCloud<pcl::PointXYZI>(m_cloud_combined, intensity_distributionCombinede, "m_cloud_combined");
    vizCombined.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "m_cloud_combined");
    //viz.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_small, normals_small, 1, 0.01);

    while (!vizC.wasStopped())
    {
        vizC.spinOnce();
    }
}

void EntropyFilterDrum::computePolyFitting(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointCloud<pcl::PointNormal> &mls_points)
{
    // Output has the PointNormal type in order to store the normals calculated by MLS

    // Create a KD-Tree
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);

    // Init object (second point type is for the normals, even if unused)
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointNormal> mls;

    mls.setComputeNormals(true);

    // Set parameters
    mls.setNumberOfThreads(4);
    mls.setInputCloud(cloud);
    mls.setPolynomialOrder(5);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(0.03);

    // Reconstruct
    mls.process(mls_points);
}

void EntropyFilterDrum::divideCloudNormals(pcl::PointCloud<pcl::PointNormal> &input, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                           pcl::PointCloud<pcl::Normal>::Ptr &normals)
{
    cloud->height = input.height;
    cloud->width = input.width;
    cloud->resize(input.size());
    normals->height = input.height;
    normals->width = input.width;
    normals->resize(input.size());
    for (int i = 0; i < input.size(); i++)
    {
        cloud->points[i].x = input.points[i].x;
        cloud->points[i].y = input.points[i].y;
        cloud->points[i].z = input.points[i].z;

        normals->points[i].normal_x = input.points[i].normal_x;
        normals->points[i].normal_y = input.points[i].normal_y;
        normals->points[i].normal_z = input.points[i].normal_z;
        normals->points[i].curvature = input.points[i].curvature;
    }
}

void EntropyFilterDrum::getSpherical(pcl::PointCloud<pcl::Normal>::Ptr &cloud_normals, pcl::PointCloud<Spherical>::Ptr &spherical)
{
    spherical->width = cloud_normals->width;
    spherical->height = cloud_normals->height;
    spherical->resize(spherical->width * spherical->height);

    pcl::Normal data;
    for (size_t i = 0; i < cloud_normals->points.size(); ++i)
    {
        data = cloud_normals->points[i];
        spherical->points[i].azimuth = atan2(data.normal_z, data.normal_y);
        spherical->points[i].inclination = atan2(sqrt(data.normal_y * data.normal_y + data.normal_z * data.normal_z), data.normal_x);
    }
}

void EntropyFilterDrum::normalizeEntropy(pcl::PointCloud<Spherical>::Ptr &spherical)
{

    float max_entropy = 0;
    for (int i = 0; i < spherical->size(); i++)
    {
        if (spherical->points[i].entropy > max_entropy)
            max_entropy = spherical->points[i].entropy;
    }

    for (int i = 0; i < spherical->size(); i++)
    {
        spherical->points[i].entropy_normalized = spherical->points[i].entropy / max_entropy;
    }
    std::cout << "max entropy : " << max_entropy << std::endl;
    _max_entropy = max_entropy;
}

//LOCAL HISTOGRAM and entropy calculation at the end.
//param[in]: point cloud normals in spherical coordinates
//param[in]: current point index in the cloud
//param[in]: vector of indeces of neighborhood points of considered on
void EntropyFilterDrum::histogram2D(pcl::PointCloud<Spherical>::Ptr &spherical, int id0, std::vector<int> indices)
{
    int Hist[64][64] = {0};
    float step_inc = M_PI / 63;
    float step_az = (2 * M_PI) / 63;
    long bin_inc, bin_az;
    for (int i = 0; i < indices.size(); i++)
    {
        bin_inc = std::lroundf(spherical->points[indices[i]].inclination / step_inc);
        bin_az = std::lroundf((spherical->points[indices[i]].azimuth + M_PI) / step_az);
        if (bin_az < 0)
            std::cout << "erorr bin_az negative" << std::endl;

        Hist[bin_inc][bin_az]++;
    }
    bin_inc = std::lroundf(spherical->points[id0].inclination / step_inc);
    bin_az = std::lroundf((spherical->points[id0].azimuth + M_PI) / step_az);
    if (bin_az < 0)
        std::cout << "erorr bin_az negative" << std::endl;

    Hist[bin_inc][bin_az]++;

    float HistNorm[64][64] = {0};
    float max_value = 0;
    for (int i = 0; i < 64; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            if (Hist[i][j] > max_value)
                max_value = Hist[i][j];
        }
    }

    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++)
            HistNorm[i][j] = Hist[i][j] / max_value;

    //entropy calculation
    float entropy_value = 0;
    float temp_entropy = 0;
    for (int i = 0; i < 64; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            temp_entropy = HistNorm[i][j] * log2(HistNorm[i][j]);
            if (!std::isnan(temp_entropy))
                entropy_value += temp_entropy;
        }
    }

    spherical->points[id0].entropy = -(entropy_value);
    //std::cout << "entropy value: " << spherical->points[id0].entropy << std::endl;
}

// LOCAL SEARCH
void EntropyFilterDrum::local_search(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<Spherical>::Ptr &spherical,
                                     pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_combined)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);
    pcl::PointXYZ searchPoint;

    for (int it = 0; it < cloud->points.size(); it++)
    {
        if (cloud_combined->points[it].intensity > 0)
        {
            searchPoint.x = cloud->points[it].x;
            searchPoint.y = cloud->points[it].y;
            searchPoint.z = cloud->points[it].z;

            // K nearest neighbor search
            std::vector<int> pointIdxNKNSearch(m_KNN);
            std::vector<float> pointNKNSquaredDistance(m_KNN);

            if (kdtree.nearestKSearch(searchPoint, m_KNN, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                histogram2D(spherical, it, pointIdxNKNSearch);
                spherical->points[it].entropy *= cloud_combined->points[it].intensity;
            }
        }
        else
        {
            spherical->points[it].entropy = 0;
        }
    }
}

void EntropyFilterDrum::segmentCloudEntropy(pcl::PointCloud<pcl::PointNormal> &cloud, pcl::PointCloud<Spherical>::Ptr &spherical,
                                            pcl::PointCloud<pcl::PointXYZI>::Ptr &output, float thresholdEntropy)
{
    pcl::PointXYZI p;
    for (int i = 0; i < spherical->size(); i++)
    {
        if (spherical->points[i].entropy_normalized > thresholdEntropy)
        {
            p.x = cloud.points[i].x;
            p.y = cloud.points[i].y;
            p.z = cloud.points[i].z;
            p.intensity = spherical->points[i].entropy_normalized;
            output->points.push_back(p);
        }
    }
    std::cout << "cloud segmented size: " << output->size() << std::endl;
}

void EntropyFilterDrum::connectedComponets(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                           std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cloud_clusters)
{
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(0.01); // 2cm
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZI>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(cloud->points[*pit]); //*

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_xyz(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*cloud_cluster, *cloud_cluster_xyz);
        cloud_clusters.push_back(cloud_cluster_xyz);
    }

    std::cout << "number of clusters found: " << cluster_indices.size() << std::endl;
}

void EntropyFilterDrum::alternativeConnectedComponets(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                                      std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cloud_clusters)
{

    for (int j = 0; j < cloud->size(); j++)
    {
        float max = 0;
        int it;
        for (int i = 0; i < cloud->size(); i++)
        {
            if (cloud->points[i].intensity > max)
            {
                max = cloud->points[i].intensity;
                it = i;
            }
        }
        //std::cout << "max entropy: " << max << ", at indices: " << it << std::endl;

        pcl::PointXYZI searchPoint = cloud->points[it];

        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
        tree->setInputCloud(cloud);
        pcl::Indices indices;
        std::vector<float> sqrDistances;
        float radius = 0.05;

        if (tree->radiusSearch(searchPoint, radius, indices, sqrDistances) > 0)
        {
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            inliers->indices = indices;
            //std::cout << "number of neighborhood: " << indices.size() << std::endl;

            pcl::ExtractIndices<pcl::PointXYZI> extract;

            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp2(new pcl::PointCloud<pcl::PointXYZ>);

            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*cloud_temp);

            pcl::copyPointCloud(*cloud_temp, *cloud_temp2);
            cloud_clusters.push_back(cloud_temp2);

            //std::cout << "added to clusters" << std::endl;

            // remove points extracted above
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloud);

            //std::cout << "remaning cloud size: " << cloud->size() << ", iteration: " << j << std::endl;
        }
    }

    std::cout << "number of clusters: " << cloud_clusters.size() << std::endl;
}

void EntropyFilterDrum::splitPointNormal(pcl::PointCloud<pcl::PointNormal>::Ptr &input, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                         pcl::PointCloud<pcl::Normal>::Ptr &normals)
{
    cloud->height = input->height;
    cloud->width = input->width;
    cloud->resize(input->size());
    normals->height = input->height;
    normals->width = input->width;
    normals->resize(input->size());
    for (int i = 0; i < input->size(); i++)
    {
        cloud->points[i].x = input->points[i].x;
        cloud->points[i].y = input->points[i].y;
        cloud->points[i].z = input->points[i].z;

        normals->points[i].normal_x = input->points[i].normal_x;
        normals->points[i].normal_y = input->points[i].normal_y;
        normals->points[i].normal_z = input->points[i].normal_z;
        normals->points[i].curvature = input->points[i].curvature;
    }
}

void EntropyFilterDrum::computeDepthMap(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                        pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_out,
                                        pcl::ModelCoefficients &line)
{
    cloud_out->width = cloud->width;
    cloud_out->height = cloud->height;
    cloud_out->resize(cloud_out->width * cloud_out->height);

    Eigen::Vector4f line_pt, line_dir;
    double sqr_norm, value;

    std::vector<std::vector<int>> idx(4);
    line_pt[0] = line.values[0];
    line_pt[1] = line.values[1];
    line_pt[2] = line.values[2];
    line_pt[3] = 0;

    line_dir[0] = line.values[3];
    line_dir[1] = line.values[4];
    line_dir[2] = line.values[5];
    line_dir[3] = 0;

    sqr_norm = sqrt(line_dir.norm());

    for (int k = 0; k < cloud->size(); k++)
    {
        value = pcl::sqrPointToLineDistance(cloud->points[k].getVector4fMap(), line_pt, line_dir, sqr_norm);

        cloud_out->points[k].x = cloud->points[k].x;
        cloud_out->points[k].y = cloud->points[k].y;
        cloud_out->points[k].z = cloud->points[k].z;

        cloud_out->points[k].intensity = sqrt(value);
    }
}

void EntropyFilterDrum::combineDepthAndCurvatureInfo(pcl::PointCloud<pcl::PointXYZI>::Ptr &depth,
                                                     pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                                     pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map)
{
    if (depth->size() != normals->size())
        PCL_WARN("Depth and Normal Clouds are of different size!\n");

    cloud_map->width = depth->width;
    cloud_map->height = depth->height;
    cloud_map->resize(depth->width * depth->height);

    std::vector<bool> result;
    result.resize(normals->size());
    float value;
    for (int i = 0; i < normals->size(); i++)
    {
        if (normals->points[i].curvature >= m_curvature_threshold && depth->points[i].intensity <= m_depth_interval)
        {
            value = (m_depth_interval - depth->points[i].intensity) * 50;
            cloud_map->points[i].intensity = value;
        }
        else
        {
            cloud_map->points[i].intensity = 0;
        }

        cloud_map->points[i].x = depth->points[i].x;
        cloud_map->points[i].y = depth->points[i].y;
        cloud_map->points[i].z = depth->points[i].z;
    }
}

//
//ColorMap functions
void EntropyFilterDrum::colorMapEntropy(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map)
{
    cloud_map->width = m_mls_cloud->width;
    cloud_map->height = m_mls_cloud->height;
    cloud_map->resize(m_mls_cloud->width * m_mls_cloud->height);

    for (int i = 0; i < m_mls_cloud->size(); i++)
    {
        cloud_map->points[i].x = m_mls_cloud->points[i].x;
        cloud_map->points[i].y = m_mls_cloud->points[i].y;
        cloud_map->points[i].z = m_mls_cloud->points[i].z;
        cloud_map->points[i].intensity = m_spherical->points[i].entropy;
    }
}

void EntropyFilterDrum::colorMapCurvature(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map)
{
    cloud_map->width = m_mls_cloud->width;
    cloud_map->height = m_mls_cloud->height;
    cloud_map->resize(m_mls_cloud->width * m_mls_cloud->height);

    for (int i = 0; i < m_mls_cloud->size(); i++)
    {
        cloud_map->points[i].x = m_mls_cloud->points[i].x;
        cloud_map->points[i].y = m_mls_cloud->points[i].y;
        cloud_map->points[i].z = m_mls_cloud->points[i].z;
        cloud_map->points[i].intensity = m_mls_normals->points[i].curvature;
    }
}

void EntropyFilterDrum::colorMapInclination(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map)
{
    cloud_map->width = m_mls_cloud->width;
    cloud_map->height = m_mls_cloud->height;
    cloud_map->resize(m_mls_cloud->width * m_mls_cloud->height);

    for (int i = 0; i < m_mls_cloud->size(); i++)
    {
        cloud_map->points[i].x = m_mls_cloud->points[i].x;
        cloud_map->points[i].y = m_mls_cloud->points[i].y;
        cloud_map->points[i].z = m_mls_cloud->points[i].z;
        cloud_map->points[i].intensity = m_spherical->points[i].inclination;
    }
}

void EntropyFilterDrum::colorMapAzimuth(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map)
{
    cloud_map->width = m_mls_cloud->width;
    cloud_map->height = m_mls_cloud->height;
    cloud_map->resize(m_mls_cloud->width * m_mls_cloud->height);

    for (int i = 0; i < m_mls_cloud->size(); i++)
    {
        cloud_map->points[i].x = m_mls_cloud->points[i].x;
        cloud_map->points[i].y = m_mls_cloud->points[i].y;
        cloud_map->points[i].z = m_mls_cloud->points[i].z;
        cloud_map->points[i].intensity = m_spherical->points[i].azimuth;
    }
}

void EntropyFilterDrum::optimizeNumberOfClouds(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cloud_clusters,
                                               std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clouds_out)
{

    std::vector<pcl::PointXYZ> results(cloud_clusters.size());
    for (int i = 0; i < cloud_clusters.size(); i++)
    {
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud_clusters[i], centroid);
        results[i].getVector4fMap() = centroid;
    }

    std::vector<int> indices(results.size());
    for (int i = 0; i < results.size(); i++)
    {
        for (int j = 0; j < results.size(); j++)
        {
            if (i == j)
                continue;

            if (pcl::geometry::distance(results[i], results[j]) < 0.05)
            {
                indices[i] = j;
                break;
            }
        }
    }

    for (int i = 0; i < indices.size(); i++)
    {
        int id = indices[i];
        if (id > 0)
        {
            for (int k = 0; k < cloud_clusters[indices[i]]->size(); k++)
                cloud_clusters[i]->push_back(cloud_clusters[indices[i]]->points[k]);

            indices[id] = -1;
        }
        if (id < 0)
            continue;

        clouds_out.push_back(cloud_clusters[i]);
    }

    std::cout << std::endl;
    std::cout << "new clusters after optimization: " << std::endl;
    for (int k = 0; k < clouds_out.size(); k++)
        std::cout
            << "new cluster " << k << ", size: " << clouds_out[k]->size() << std::endl;
}
