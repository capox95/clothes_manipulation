#include "../include/drumModel.h"

// INPUT
void DrumModel::setInputCloud(pcl::PointCloud<pcl::PointNormal>::Ptr &input) { _input = input; }

void DrumModel::setDrumAxis(Eigen::Vector3f &axis_pt, Eigen::Vector3f &axis_dir)
{
    _axis.values = {axis_pt.x(), axis_pt.y(), axis_pt.z(), axis_dir.x(), axis_dir.y(), axis_dir.z()};
}

void DrumModel::setDrumCenterDistance(float distance) { _distanceCenter = distance; }

void DrumModel::setDrumRadius(float radius) { _radius = radius; }

void DrumModel::visualize(pcl::PointCloud<pcl::PointNormal>::Ptr &scene,
                          bool planes_flag, bool cylinder_flag, bool lines_flag)
{
    // Visualization
    pcl::visualization::PCLVisualizer vizS("PCL");
    vizS.addCoordinateSystem(0.1, "coordinate");
    vizS.setBackgroundColor(1.0, 1.0, 1.0);
    vizS.addPointCloud<pcl::PointNormal>(scene, "source");
    vizS.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.7, 0.0, "source");

    if (planes_flag)
    {
        vizS.addPlane(_planes[0], "planes0");
        vizS.addPlane(_planes[1], "planes1");
    }

    if (lines_flag)
    {
        pcl::PointXYZ line1_0 = pcl::PointXYZ(_axis.values[0], _axis.values[1], _axis.values[2]);

        pcl::PointXYZ line1_1 = pcl::PointXYZ(line1_0.x + 0.75 * _axis.values[3],
                                              line1_0.y + 0.75 * _axis.values[4],
                                              line1_0.z + 0.75 * _axis.values[5]);
        vizS.addLine(line1_0, line1_1, 1.0f, 0.0f, 0.0f, "line1");
    }

    vizS.addSphere(_maxPaddlePoint, 0.01, 1.0f, 0.0f, 0.0f, "_maxPaddlePoint");
    vizS.addSphere(_minPaddlePoint, 0.01, 1.0f, 0.0f, 0.0f, "_minPaddlePoint");
    vizS.addSphere(_centerPaddleProjected, 0.01, 1.0f, 1.0f, 0.0f, "_centerPaddleProjected");

    vizS.addSphere(_centerPaddlePoint, 0.01, 1.0f, 1.0f, 0.0f, "_centerPaddlePoint");
    vizS.addSphere(_centerPaddle2Point, 0.01, 1.0f, 1.0f, 0.0f, "_centerPaddle2Point");
    vizS.addSphere(_centerPaddle3Point, 0.01, 1.0f, 1.0f, 0.0f, "_centerPaddle3Point");

    vizS.addSphere(_paddlesCenter[0], 0.01, 0.0f, 0.0f, 1.0f, "_paddlesCenter0");
    vizS.addSphere(_paddlesCenter[1], 0.01, 0.0f, 0.0f, 1.0f, "_paddlesCenter1");
    vizS.addSphere(_paddlesCenter[2], 0.01, 0.0f, 0.0f, 1.0f, "_paddlesCenter2");

    if (cylinder_flag)
        vizS.addCylinder(_cylinder, "cylinder");

    vizS.spin();
}

void DrumModel::compute(Affine3dVector &transformation_matrix_vector)
{
    // --------------------------------------
    //pcl::ModelCoefficients line1, line2;
    //computeDrumAxes(line1, line2);
    // --------------------------------------
    //

    _axis_dir = {_axis.values[3], _axis.values[4], _axis.values[5]};
    _origin = {_axis.values[0], _axis.values[1], _axis.values[2]};
    _centerPoint.getVector3fMap() = _origin + _axis_dir * _distanceCenter;

    // variable made of axis defined with respect center of the drum
    _axisDrum.values = {_centerPoint.x, _centerPoint.y, _centerPoint.z, _axis.values[3], _axis.values[4], _axis.values[5]};

    // work on first paddle to find line and verify that is parallel to axis of drum
    findPlanes(_input, _planes);
    estimateIntersactionLine(_planes, _line);
    checkIfParallel(_line, _axisDrum);

    // calculate center, max, min points on paddle line
    getPointsOnLine(_input, _line, _maxPaddlePoint, _minPaddlePoint, _centerPaddlePoint);

    //projection of centerFin point on axis line of drum
    _centerPaddleProjected = projection(_centerPaddlePoint, _axisDrum);

    // check if distance between paddle center and corresponding point on axis is about radius of drum
    float distance = pcl::euclideanDistance(_centerPaddlePoint, _centerPaddleProjected);
    std::cout << "distance paddle - drum axis: " << distance << std::endl;
    if (distance > _radius)
    {
        PCL_WARN("distance paddle-drum axis larger than radius of the drum.\n");
    }

    calculatePaddleHeight(_input, _line, _paddleHeight);

    calculateNewPoints(_axisDrum, _centerPaddleProjected, _centerPaddlePoint, _centerPaddle2Point, _centerPaddle3Point);

    std::vector<pcl::PointXYZ> Points{_centerPaddlePoint, _centerPaddle2Point, _centerPaddle3Point};
    _paddlesCenter = movePointsToPaddlesCenter(Points, _centerPaddleProjected, _paddleHeight);

    _cylinder.values = {_centerPoint.x, _centerPoint.y, _centerPoint.z,
                        _axis_dir.x(), _axis_dir.y(), _axis_dir.z(), distance};

    computeTransformation(transformation_matrix_vector);
}

// estimate the planes for the two surfaces of the fin model.
// The intersection of the two planes will give us the line of the top part of the fin.
// planes is a vector of 2 elements
void DrumModel::findPlanes(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_plane, std::vector<pcl::ModelCoefficients> &planes)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud(*cloud_plane, *cloud_copy);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointNormal> seg;
    pcl::ModelCoefficients plane;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    for (int i = 0; i < 2; i++)
    {
        seg.setInputCloud(cloud_copy);
        seg.segment(*inliers, plane);

        if (inliers->indices.size() == 0)
        {
            std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        }
        pcl::console::print_highlight("inliers size: %d \n", inliers->indices.size());

        pcl::ExtractIndices<pcl::PointNormal> extract;
        // Extract the inliers
        extract.setInputCloud(cloud_copy);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_copy);

        planes.push_back(plane);
    }
}

// estimate the intersaction line resulting from the two planes.
void DrumModel::estimateIntersactionLine(std::vector<pcl::ModelCoefficients> &planes, pcl::ModelCoefficients &line_model)
{
    Eigen::Vector4f plane_a(planes[0].values.data());
    Eigen::Vector4f plane_b(planes[1].values.data());

    if (plane_a.w() > 0)
        plane_a = -plane_a;

    if (plane_b.w() > 0)
        plane_b = -plane_b;

    float dot_product = acos(plane_a.head<3>().dot(plane_b.head<3>())); // can be used to verify planes estimations
    if (dot_product > 2.2 || dot_product < 1.9)
        pcl::console::print_highlight("angle between planes is not consistent! value: %f\n", dot_product);

    Eigen::VectorXf line;
    pcl::planeWithPlaneIntersection(plane_a, plane_b, line);

    std::vector<float> values(&line[0], line.data() + line.cols() * line.rows());
    line_model.values = values;
}

void DrumModel::getPointsOnLine(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud, pcl::ModelCoefficients &line,
                                pcl::PointXYZ &maxFinPoint, pcl::PointXYZ &minFinPoint, pcl::PointXYZ &centerFinPoint)
{
    pcl::PointXYZ line_point = pcl::PointXYZ(line.values[0], line.values[1], line.values[2]);

    std::vector<float> distances;
    for (int i = 0; i < cloud->size(); i++)
    {
        distances.push_back(pcl::euclideanDistance<pcl::PointXYZ, pcl::PointNormal>(line_point, cloud->points[i]));
    }

    int maxIdx = std::max_element(distances.begin(), distances.end()) - distances.begin();
    int minIdx = std::min_element(distances.begin(), distances.end()) - distances.begin();

    pcl::PointXYZ maxPoint, minPoint;

    maxPoint.x = cloud->points[maxIdx].x;
    maxPoint.y = cloud->points[maxIdx].y;
    maxPoint.z = cloud->points[maxIdx].z;

    minPoint.x = cloud->points[minIdx].x;
    minPoint.y = cloud->points[minIdx].y;
    minPoint.z = cloud->points[minIdx].z;

    Eigen::Vector3f line_pt, line_dir;
    line_pt = line_point.getVector3fMap();
    line_dir = {line.values[3], line.values[4], line.values[5]};

    /*
    //project points onto the line
    Eigen::Vector3f minP, maxP;
    minP = minPoint.getVector3fMap() - line_pt;
    minPoint.getVector3fMap() = line_pt + minP.dot(line_dir) * line_dir;

    maxP = maxPoint.getVector3fMap() - line_pt;
    maxPoint.getVector3fMap() = line_pt + maxP.dot(line_dir) * line_dir;

    std::cout << maxPoint << ", " << minPoint << std::endl;
    */

    //project points onto the line
    Eigen::Vector3f LL{line.values[3], line.values[4], line.values[5]};

    Eigen::Vector3f minP, maxP;
    minP = minPoint.getVector3fMap() - line_point.getVector3fMap();
    maxP = maxPoint.getVector3fMap() - line_point.getVector3fMap();

    Eigen::Vector3f minPointProjected, maxPointProjected;
    minPointProjected = minP.dot(LL) / LL.norm() * LL;
    minPointProjected.normalize();
    minPoint.getVector3fMap() = line_point.getVector3fMap() + minPointProjected * minP.norm();

    maxPointProjected = maxP.dot(LL) / LL.norm() * LL;
    maxPointProjected.normalize();
    maxPoint.getVector3fMap() = line_point.getVector3fMap() + maxPointProjected * maxP.norm();

    _paddleLength = pcl::euclideanDistance<pcl::PointXYZ, pcl::PointXYZ>(maxPoint, minPoint);
    pcl::console::print_highlight("Fin estimated length: %f meters\n", _paddleLength);

    pcl::PointXYZ midPoint;
    midPoint.x = (maxPoint.x + minPoint.x) / 2;
    midPoint.y = (maxPoint.y + minPoint.y) / 2;
    midPoint.z = (maxPoint.z + minPoint.z) / 2;

    //modify center point
    //std::cout << "Old Line center: " << line.values[0] << ", " << line.values[1] << ", " << line.values[2] << std::endl;
    line.values[0] = midPoint.x;
    line.values[1] = midPoint.y;
    line.values[2] = midPoint.z;
    //std::cout << "New Line center: " << line.values[0] << ", " << line.values[1] << ", " << line.values[2] << std::endl;

    maxFinPoint = maxPoint;
    minFinPoint = minPoint;
    centerFinPoint = midPoint;
}

//to determine if two lines are parallel in 3D
bool DrumModel::checkIfParallel(pcl::ModelCoefficients &line, pcl::ModelCoefficients &axis)
{
    // normalizing the vectors to unit length and computing the norm of the cross-product,
    // which is the sine of the angle between them.

    Eigen::Vector3f vec1;
    vec1.x() = line.values[3];
    vec1.y() = line.values[4];
    vec1.z() = line.values[5];
    vec1.normalize();

    Eigen::Vector3f vec2;
    vec2.x() = axis.values[3];
    vec2.y() = axis.values[4];
    vec2.z() = axis.values[5];
    vec2.normalize();

    Eigen::Vector3f vecCross = vec1.cross(vec2);
    float norm = vecCross.norm();
    if (abs(asin(norm)) > 0.1)
    {
        PCL_WARN("Axes are not parallel!\n");
        return false;
    }

    return true;
}

pcl::PointXYZ DrumModel::projection(pcl::PointXYZ &point, pcl::ModelCoefficients &line)
{
    pcl::PointXYZ linePoint;
    linePoint.x = line.values[0];
    linePoint.y = line.values[1];
    linePoint.z = line.values[2];

    //project points onto the line
    Eigen::Vector3f V;
    V = point.getVector3fMap() - linePoint.getVector3fMap();

    Eigen::Vector3f L{line.values[3], line.values[4], line.values[5]};

    Eigen::Vector3f projectedV;
    projectedV = linePoint.getVector3fMap() + V.dot(L) / L.dot(L) * L;

    pcl::PointXYZ projectedPoint;
    projectedPoint.getVector3fMap() = projectedV;

    //check if projected point is near real center of the drum
    Eigen::Vector3f center = linePoint.getVector3fMap();

    if ((center - projectedV).norm() > 0.1)
    {
        PCL_WARN("center drum and projected points not close enough!\n");
    }

    return projectedPoint;
}

void DrumModel::calculateNewPoints(pcl::ModelCoefficients &cylinder, pcl::PointXYZ &centerCylinder,
                                   pcl::PointXYZ &centerFin1, pcl::PointXYZ &centerFin2, pcl::PointXYZ &centerFin3)
{

    std::vector<pcl::PointXYZ> newPoints;
    pcl::PointXYZ new_point;
    // Rodrigues' rotation formula

    // angle to rotate
    float theta = (2 * M_PI) / 3.0f;

    // unit versor k
    Eigen::Vector3f k{cylinder.values[3], cylinder.values[4], cylinder.values[5]};
    k.normalize();

    // vector to rotate V
    Eigen::Vector3f V{centerFin1.getVector3fMap() - centerCylinder.getVector3fMap()};
    float normV = V.norm();
    Eigen::Vector3f V_rot;

    // computation of two points, each displaced by 2pi/3
    for (int c = 0; c < 2; c++)
    {

        V_rot = V * cos(theta) + (k.cross(V)) * sin(theta) + k * (k.dot(V)) * (1 - cos(theta));
        V_rot.normalize();

        new_point.x = centerCylinder.x + normV * V_rot.x();
        new_point.y = centerCylinder.y + normV * V_rot.y();
        new_point.z = centerCylinder.z + normV * V_rot.z();

        newPoints.push_back(new_point);
        V = V_rot;
    }

    centerFin2 = newPoints[0];
    centerFin3 = newPoints[1];
}

std::vector<pcl::PointXYZ> DrumModel::movePointsToPaddlesCenter(std::vector<pcl::PointXYZ> &points, pcl::PointXYZ &center, float height)
{
    float distance = 0.5 * height;

    std::vector<pcl::PointXYZ> result;
    for (int i = 0; i < points.size(); i++)
    {
        //calculate vector from center to point and normalize it to get versor
        Eigen::Vector3f vec = points[i].getVector3fMap() - center.getVector3fMap();
        vec.normalize();

        //compute new point along versor direction
        pcl::PointXYZ newPoint;
        newPoint.getVector3fMap() = points[i].getVector3fMap() + vec * distance;

        result.push_back(newPoint);
    }
    return result;
}

float DrumModel::calculatePaddleHeight(pcl::PointCloud<pcl::PointNormal>::Ptr &input, pcl::ModelCoefficients &line, float &height)
{
    std::vector<float> distances;
    float tmp;

    Eigen::Vector4f line_pt(line.values[0], line.values[1], line.values[2], 0);
    Eigen::Vector4f line_dir(line.values[3], line.values[4], line.values[5], 0);
    double sqr = line_dir.norm() * line_dir.norm();

    for (int i = 0; i < input->size(); i++)
    {
        tmp = pcl::sqrPointToLineDistance(input->points[i].getVector4fMap(), line_pt, line_dir, sqr);
        distances.push_back(tmp);
    }

    double max = *std::max_element(distances.begin(), distances.end());
    height = sqrt(max);

    pcl::console::print_highlight("Fin estimated height: %f meters\n", height);
}

void DrumModel::computeTransformation(Affine3dVector &transformation_matrix_vector)
{
    Eigen::Vector3f axis2 = _centerPaddlePoint.getVector3fMap() - _centerPaddleProjected.getVector3fMap();
    axis2.normalize();

    Eigen::Vector3f axis1 = _axis_dir;
    axis1.normalize();

    Eigen::VectorXd from_line_x, from_line_z, to_line_x, to_line_z;

    from_line_x.resize(6);
    from_line_z.resize(6);
    to_line_x.resize(6);
    to_line_z.resize(6);

    //Origin
    from_line_x << 0, 0, 0, 1, 0, 0;
    from_line_z << 0, 0, 0, 0, 0, 1;

    to_line_x.head<3>() = _centerPaddleProjected.getVector3fMap().cast<double>();
    to_line_x.tail<3>() = axis2.cast<double>();

    to_line_z.head<3>() = _centerPaddleProjected.getVector3fMap().cast<double>();
    to_line_z.tail<3>() = axis1.cast<double>();

    Eigen::Affine3d transformation;
    if (!(pcl::transformBetween2CoordinateSystems(from_line_x, from_line_z, to_line_x, to_line_z, transformation)))
        std::cout << "error computing affine transform" << std::endl;

    // each cylinder (paddle) is defined by its own affine transformation matrix
    Eigen::Affine3d tmp;
    tmp.linear() = transformation.linear();

    for (int i = 0; i < _paddlesCenter.size(); i++)
    {
        tmp.translation() = _paddlesCenter[i].getVector3fMap().cast<double>();
        transformation_matrix_vector.push_back(tmp);
    }
}

//--------------------------------------
/*
void DrumModel::computeDrumAxes(pcl::ModelCoefficients &line1, pcl::ModelCoefficients &line2)
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
    
    //line1.values[0] = 0;
    //line1.values[1] = 0.09;
    //line1.values[2] = 0.11;
    //line2.values[0] = 0;
    //line2.values[1] = 0.09;
    //line2.values[2] = 0.11;
    

    std::cout << "axis computed: -------------------------" << std::endl;
    for (int i = 0; i < line1.values.size(); i++)
    {
        std::cout << line1.values[i] << ", ";
    }
    std::cout << std::endl;

    for (int i = 0; i < line2.values.size(); i++)
    {
        std::cout << line2.values[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "-------------------------" << std::endl;
    }

void DrumModel::transformation(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
{
    float theta = -0.785398;

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << 0.0, 0.0, 0.0;
    transform.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitX()));
    pcl::transformPointCloud(*cloud, *cloud_out, transform);
}

*/