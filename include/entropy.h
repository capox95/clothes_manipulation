
struct Spherical
{
    float inclination;
    float azimuth;
    float entropy;
    float entropy_normalized;
};

class EntropyFilterDrum
{

private:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_source;
    pcl::PointCloud<pcl::PointNormal> m_mls_points;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_mls_cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_cloud_depth, m_cloud_combined, m_cloud_seg;
    pcl::PointCloud<pcl::Normal>::Ptr m_mls_normals;
    pcl::PointCloud<Spherical>::Ptr m_spherical;

    pcl::ModelCoefficients m_line;

    float m_drum_radius, m_entropy_threshold, m_curvature_threshold, m_depth_interval, m_depth_threshold, _max_entropy;
    int m_KNN;
    bool _flag_vertices, _flag_optimization;

public:
    EntropyFilterDrum() : m_source(new pcl::PointCloud<pcl::PointXYZRGB>),
                          m_mls_cloud(new pcl::PointCloud<pcl::PointXYZ>),
                          m_mls_normals(new pcl::PointCloud<pcl::Normal>),
                          m_spherical(new pcl::PointCloud<Spherical>),
                          m_cloud_seg(new pcl::PointCloud<pcl::PointXYZI>),
                          m_cloud_depth(new pcl::PointCloud<pcl::PointXYZI>),
                          m_cloud_combined(new pcl::PointCloud<pcl::PointXYZI>)

    {
        _flag_vertices = false;
        _flag_optimization = false;
    }
    void optimizeNumberOfClouds(bool flag) { _flag_optimization = flag; }

    void setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in);

    void setDrumAxis(pcl::ModelCoefficients line);

    void setDownsampleLeafSize(float leaf_size);

    void setEntropyThreshold(float entropy_th);

    void setKLocalSearch(int K);

    void setCurvatureThreshold(float curvature_th);

    void setDepthThreshold(float depth_th);

    void setDrumRadius(float radius);

    bool compute(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clouds_out);

    //
    //ColorMap functions
    void colorMapEntropy(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map);

    void colorMapCurvature(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map);

    void colorMapInclination(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map);

    void colorMapAzimuth(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map);

    void visualizeAll(bool flag);

private:
    void computePolyFitting(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointCloud<pcl::PointNormal> &mls_points);

    void divideCloudNormals(pcl::PointCloud<pcl::PointNormal> &input, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                            pcl::PointCloud<pcl::Normal>::Ptr &normals);

    void getSpherical(pcl::PointCloud<pcl::Normal>::Ptr &cloud_normals, pcl::PointCloud<Spherical>::Ptr &spherical);

    void normalizeEntropy(pcl::PointCloud<Spherical>::Ptr &spherical);

    //LOCAL HISTOGRAM and entropy calculation at the end.
    //param[in]: point cloud normals in spherical coordinates
    //param[in]: current point index in the cloud
    //param[in]: vector of indeces of neighborhood points of considered on
    void histogram2D(pcl::PointCloud<Spherical>::Ptr &spherical, int id0, std::vector<int> indices);

    // LOCAL SEARCH
    void local_search(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<Spherical>::Ptr &spherical,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_combined);

    void segmentCloudEntropy(pcl::PointCloud<pcl::PointNormal> &cloud, pcl::PointCloud<Spherical>::Ptr &spherical,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr &output, float thresholdEntropy);

    void connectedComponets(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cloud_clusters);

    void alternativeConnectedComponets(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                       std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cloud_clusters);

    void splitPointNormal(pcl::PointCloud<pcl::PointNormal>::Ptr &input,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                          pcl::PointCloud<pcl::Normal>::Ptr &normals);

    //--------------------------------------------------------------------------------------------------------------
    void computeDepthMap(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_out,
                         pcl::ModelCoefficients &line);

    void combineDepthAndCurvatureInfo(pcl::PointCloud<pcl::PointXYZI>::Ptr &depth,
                                      pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                      pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_map);

    void optimizeNumberOfClouds(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cloud_clusters,
                                std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clouds_out);
};
