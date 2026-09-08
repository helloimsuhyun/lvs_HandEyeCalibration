#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/pcl_config.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot.h>
#include <pcl/features/ppf.h>
#include <pcl/features/pfh_tools.h>

#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/sift_keypoint.h>

#include <pcl/registration/ppf_registration.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

using ArrayF = py::array_t<float, py::array::c_style | py::array::forcecast>;

template <typename T>
static py::array_t<T> make_array(std::initializer_list<py::ssize_t> shape) {
    // Explicit std::vector avoids pybind11 3.x initializer-list ambiguity.
    return py::array_t<T>(std::vector<py::ssize_t>(shape));
}

static void require_positive(float x, const char* name) {
    if (!(x > 0.0f) || !std::isfinite(x)) {
        throw std::invalid_argument(std::string(name) + " must be finite and > 0");
    }
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud_from_numpy(const ArrayF& points) {
    auto b = points.request();
    if (b.ndim != 2 || b.shape[1] != 3) {
        throw std::invalid_argument("points must have shape (N, 3)");
    }
    const auto n = static_cast<std::size_t>(b.shape[0]);
    const float* p = static_cast<const float*>(b.ptr);

    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->resize(n);
    cloud->width = static_cast<std::uint32_t>(n);
    cloud->height = 1;
    cloud->is_dense = true;

    for (std::size_t i = 0; i < n; ++i) {
        const float x = p[3*i + 0];
        const float y = p[3*i + 1];
        const float z = p[3*i + 2];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            throw std::invalid_argument("points contain NaN/Inf");
        }
        (*cloud)[i].x = x;
        (*cloud)[i].y = y;
        (*cloud)[i].z = z;
    }
    return cloud;
}

static pcl::PointCloud<pcl::Normal>::Ptr normal_cloud_from_numpy(
    const ArrayF& normals,
    std::size_t expected_n
) {
    auto b = normals.request();
    if (b.ndim != 2 || b.shape[1] != 3 || static_cast<std::size_t>(b.shape[0]) != expected_n) {
        throw std::invalid_argument("normals must have shape (N, 3) matching points");
    }
    const float* p = static_cast<const float*>(b.ptr);

    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();
    cloud->resize(expected_n);
    cloud->width = static_cast<std::uint32_t>(expected_n);
    cloud->height = 1;

    for (std::size_t i = 0; i < expected_n; ++i) {
        Eigen::Vector3f n(p[3*i + 0], p[3*i + 1], p[3*i + 2]);
        if (!n.allFinite() || n.norm() < 1e-12f) {
            throw std::invalid_argument("normals contain NaN/Inf or zero-length vectors");
        }
        n.normalize();
        (*cloud)[i].normal_x = n.x();
        (*cloud)[i].normal_y = n.y();
        (*cloud)[i].normal_z = n.z();
    }
    cloud->is_dense = true;
    return cloud;
}

static pcl::PointCloud<pcl::Normal>::Ptr estimate_normals_impl(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
    float radius,
    py::object viewpoint,
    int threads
) {
    require_positive(radius, "normal radius");
    auto normals = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne(
        static_cast<unsigned int>(std::max(0, threads))
    );
    ne.setInputCloud(cloud);
    ne.setRadiusSearch(radius);

    if (!viewpoint.is_none()) {
        ArrayF vp = py::cast<ArrayF>(viewpoint);
        auto b = vp.request();
        if (b.size != 3) {
            throw std::invalid_argument("viewpoint must contain exactly 3 values");
        }
        const float* v = static_cast<const float*>(b.ptr);
        ne.setViewPoint(v[0], v[1], v[2]);
    }
    ne.compute(*normals);
    return normals;
}

static pcl::PointCloud<pcl::Normal>::Ptr normals_or_estimate(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
    py::object normals_obj,
    float normal_radius,
    py::object viewpoint,
    int threads
) {
    if (!normals_obj.is_none()) {
        return normal_cloud_from_numpy(
            py::cast<ArrayF>(normals_obj),
            cloud->size()
        );
    }
    return estimate_normals_impl(cloud, normal_radius, viewpoint, threads);
}

static pcl::PointCloud<pcl::PointNormal>::Ptr combine_xyz_normals(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& xyz,
    const pcl::PointCloud<pcl::Normal>::ConstPtr& normals
) {
    if (xyz->size() != normals->size()) {
        throw std::invalid_argument("point/normal count mismatch");
    }
    auto out = pcl::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    out->resize(xyz->size());
    out->width = xyz->width;
    out->height = xyz->height;
    out->is_dense = xyz->is_dense && normals->is_dense;

    for (std::size_t i = 0; i < xyz->size(); ++i) {
        (*out)[i].x = (*xyz)[i].x;
        (*out)[i].y = (*xyz)[i].y;
        (*out)[i].z = (*xyz)[i].z;
        (*out)[i].normal_x = (*normals)[i].normal_x;
        (*out)[i].normal_y = (*normals)[i].normal_y;
        (*out)[i].normal_z = (*normals)[i].normal_z;
        (*out)[i].curvature = (*normals)[i].curvature;
    }
    return out;
}

static py::array_t<float> xyz_to_numpy(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
    auto out = make_array<float>({static_cast<py::ssize_t>(cloud.size()), 3});
    auto a = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(cloud.size()); ++i) {
        a(i,0) = cloud[static_cast<std::size_t>(i)].x;
        a(i,1) = cloud[static_cast<std::size_t>(i)].y;
        a(i,2) = cloud[static_cast<std::size_t>(i)].z;
    }
    return out;
}

static py::dict estimate_normals_py(
    const ArrayF& points,
    float radius,
    py::object viewpoint,
    int threads
) {
    auto cloud = xyz_cloud_from_numpy(points);
    auto normals = estimate_normals_impl(cloud, radius, viewpoint, threads);

    auto n_arr = make_array<float>({static_cast<py::ssize_t>(normals->size()), 3});
    auto c_arr = make_array<float>({static_cast<py::ssize_t>(normals->size())});
    auto n = n_arr.mutable_unchecked<2>();
    auto c = c_arr.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(normals->size()); ++i) {
        const auto& q = (*normals)[static_cast<std::size_t>(i)];
        n(i,0) = q.normal_x;
        n(i,1) = q.normal_y;
        n(i,2) = q.normal_z;
        c(i) = q.curvature;
    }

    py::dict d;
    d["normals"] = n_arr;
    d["curvature"] = c_arr;
    return d;
}

static py::array_t<float> voxel_downsample_py(const ArrayF& points, float leaf_size) {
    require_positive(leaf_size, "leaf_size");
    auto cloud = xyz_cloud_from_numpy(points);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    pcl::PointCloud<pcl::PointXYZ> out;
    vg.filter(out);
    return xyz_to_numpy(out);
}

static pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>::ResponseMethod
parse_harris_method(const std::string& method) {
    using H = pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>;
    if (method == "HARRIS") return H::HARRIS;
    if (method == "NOBLE") return H::NOBLE;
    if (method == "LOWE") return H::LOWE;
    if (method == "TOMASI") return H::TOMASI;
    if (method == "CURVATURE") return H::CURVATURE;
    throw std::invalid_argument("method must be HARRIS/NOBLE/LOWE/TOMASI/CURVATURE");
}

static py::dict harris3d_py(
    const ArrayF& points,
    float radius,
    float threshold,
    bool nonmax,
    bool refine,
    const std::string& method,
    py::object normals_obj,
    int threads
) {
    require_positive(radius, "radius");
    auto cloud = xyz_cloud_from_numpy(points);

    pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> h(
        parse_harris_method(method), radius, threshold
    );
    h.setInputCloud(cloud);
    h.setRadius(radius);
    h.setThreshold(threshold);
    h.setNonMaxSupression(nonmax);
    h.setRefine(refine);
    h.setNumberOfThreads(static_cast<unsigned int>(std::max(0, threads)));

    if (!normals_obj.is_none()) {
        auto normals = normal_cloud_from_numpy(py::cast<ArrayF>(normals_obj), cloud->size());
        h.setNormals(normals);
    }

    pcl::PointCloud<pcl::PointXYZI> out;
    h.compute(out);

    auto xyz = make_array<float>({static_cast<py::ssize_t>(out.size()), 3});
    auto response = make_array<float>({static_cast<py::ssize_t>(out.size())});
    auto x = xyz.mutable_unchecked<2>();
    auto r = response.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(out.size()); ++i) {
        const auto& q = out[static_cast<std::size_t>(i)];
        x(i,0) = q.x; x(i,1) = q.y; x(i,2) = q.z;
        r(i) = q.intensity;
    }

    py::dict d;
    d["points"] = xyz;
    d["response"] = response;
    return d;
}

static py::array_t<float> iss3d_py(
    const ArrayF& points,
    float salient_radius,
    float nonmax_radius,
    float gamma21,
    float gamma32,
    int min_neighbors,
    int threads
) {
    require_positive(salient_radius, "salient_radius");
    require_positive(nonmax_radius, "nonmax_radius");
    if (!(gamma21 > 0.0f && gamma21 < 1.0f && gamma32 > 0.0f && gamma32 < 1.0f)) {
        throw std::invalid_argument("gamma21/gamma32 must be in (0,1)");
    }
    if (min_neighbors < 1) throw std::invalid_argument("min_neighbors must be >= 1");

    auto cloud = xyz_cloud_from_numpy(points);
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss;
    iss.setInputCloud(cloud);
    iss.setSalientRadius(salient_radius);
    iss.setNonMaxRadius(nonmax_radius);
    iss.setThreshold21(gamma21);
    iss.setThreshold32(gamma32);
    iss.setMinNeighbors(min_neighbors);
    iss.setNumberOfThreads(std::max(0, threads));

    pcl::PointCloud<pcl::PointXYZ> out;
    iss.compute(out);
    return xyz_to_numpy(out);
}

static py::array_t<float> sift3d_py(
    const ArrayF& points,
    const ArrayF& intensity,
    float min_scale,
    int n_octaves,
    int n_scales_per_octave,
    float min_contrast
) {
    require_positive(min_scale, "min_scale");
    if (n_octaves < 1 || n_scales_per_octave < 1) {
        throw std::invalid_argument("n_octaves and n_scales_per_octave must be >= 1");
    }

    auto pb = points.request();
    auto ib = intensity.request();
    if (pb.ndim != 2 || pb.shape[1] != 3) {
        throw std::invalid_argument("points must have shape (N,3)");
    }
    if (ib.ndim != 1 || ib.shape[0] != pb.shape[0]) {
        throw std::invalid_argument("intensity must have shape (N,)");
    }

    const float* p = static_cast<const float*>(pb.ptr);
    const float* it = static_cast<const float*>(ib.ptr);
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    cloud->resize(static_cast<std::size_t>(pb.shape[0]));

    for (std::size_t i = 0; i < cloud->size(); ++i) {
        (*cloud)[i].x = p[3*i+0];
        (*cloud)[i].y = p[3*i+1];
        (*cloud)[i].z = p[3*i+2];
        (*cloud)[i].intensity = it[i];
    }

    pcl::SIFTKeypoint<pcl::PointXYZI, pcl::PointWithScale> sift;
    sift.setInputCloud(cloud);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);

    pcl::PointCloud<pcl::PointWithScale> out;
    sift.compute(out);

    auto result = make_array<float>({static_cast<py::ssize_t>(out.size()), 4});
    auto a = result.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(out.size()); ++i) {
        const auto& q = out[static_cast<std::size_t>(i)];
        a(i,0) = q.x; a(i,1) = q.y; a(i,2) = q.z; a(i,3) = q.scale;
    }
    return result;
}

static py::array_t<float> fpfh_py(
    const ArrayF& points,
    float radius,
    py::object normals_obj,
    float normal_radius,
    py::object viewpoint,
    int threads
) {
    require_positive(radius, "FPFH radius");
    require_positive(normal_radius, "normal_radius");
    auto cloud = xyz_cloud_from_numpy(points);
    auto normals = normals_or_estimate(
        cloud, normals_obj, normal_radius, viewpoint, threads
    );

    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> est(
        static_cast<unsigned int>(std::max(0, threads))
    );
    est.setInputCloud(cloud);
    est.setInputNormals(normals);
    est.setRadiusSearch(radius);

    pcl::PointCloud<pcl::FPFHSignature33> out;
    est.compute(out);

    auto result = make_array<float>({static_cast<py::ssize_t>(out.size()), 33});
    auto a = result.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(out.size()); ++i) {
        for (int j = 0; j < 33; ++j) {
            a(i,j) = out[static_cast<std::size_t>(i)].histogram[j];
        }
    }
    return result;
}

static py::array_t<float> shot_py(
    const ArrayF& points,
    float radius,
    py::object normals_obj,
    float normal_radius,
    py::object viewpoint
) {
    require_positive(radius, "SHOT radius");
    require_positive(normal_radius, "normal_radius");
    auto cloud = xyz_cloud_from_numpy(points);
    auto normals = normals_or_estimate(
        cloud, normals_obj, normal_radius, viewpoint, 0
    );

    pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> est;
    est.setInputCloud(cloud);
    est.setInputNormals(normals);
    est.setRadiusSearch(radius);
    est.setLRFRadius(radius);

    pcl::PointCloud<pcl::SHOT352> out;
    est.compute(out);

    auto result = make_array<float>({static_cast<py::ssize_t>(out.size()), 352});
    auto a = result.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(out.size()); ++i) {
        for (int j = 0; j < 352; ++j) {
            a(i,j) = out[static_cast<std::size_t>(i)].descriptor[j];
        }
    }
    return result;
}


struct CompatPoseWithVotes {
    Eigen::Affine3f pose = Eigen::Affine3f::Identity();
    unsigned int votes = 0;
};

static bool compat_pose_within_bounds(
    const Eigen::Affine3f& a,
    const Eigen::Affine3f& b,
    float position_threshold,
    float rotation_threshold_rad
) {
    const float position_diff = (a.translation() - b.translation()).norm();
    const Eigen::AngleAxisf rotation_diff(
        a.rotation().inverse().lazyProduct(b.rotation()).eval()
    );
    const float rotation_diff_angle = std::abs(rotation_diff.angle());
    return position_diff < position_threshold &&
           rotation_diff_angle < rotation_threshold_rad;
}

static std::vector<CompatPoseWithVotes> compat_cluster_poses_pcl112(
    std::vector<CompatPoseWithVotes> poses,
    float position_threshold,
    float rotation_threshold_rad,
    int max_candidates
) {
    std::sort(
        poses.begin(), poses.end(),
        [](const CompatPoseWithVotes& a, const CompatPoseWithVotes& b) {
            return a.votes > b.votes;
        }
    );

    std::vector<std::vector<CompatPoseWithVotes>> clusters;
    std::vector<unsigned int> cluster_votes;

    // Mirrors the PCL 1.12 PPFRegistration clustering rule:
    // compare to the first pose of each existing cluster.
    for (const auto& pose : poses) {
        bool assigned = false;
        for (std::size_t c = 0; c < clusters.size(); ++c) {
            if (compat_pose_within_bounds(
                    pose.pose,
                    clusters[c].front().pose,
                    position_threshold,
                    rotation_threshold_rad)) {
                clusters[c].push_back(pose);
                cluster_votes[c] += pose.votes;
                assigned = true;
                break;
            }
        }
        if (!assigned) {
            clusters.push_back({pose});
            cluster_votes.push_back(pose.votes);
        }
    }

    std::vector<std::size_t> order(clusters.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(
        order.begin(), order.end(),
        [&](std::size_t a, std::size_t b) {
            return cluster_votes[a] > cluster_votes[b];
        }
    );

    std::vector<CompatPoseWithVotes> result;
    const std::size_t limit = std::min<std::size_t>(
        order.size(),
        static_cast<std::size_t>(std::max(1, max_candidates))
    );

    for (std::size_t rank = 0; rank < limit; ++rank) {
        const std::size_t cid = order[rank];
        const auto& cluster = clusters[cid];
        if (cluster.empty()) continue;

        Eigen::Vector3f translation_average(0.f, 0.f, 0.f);
        Eigen::Vector4f rotation_average(0.f, 0.f, 0.f, 0.f);

        // Keep the same simple quaternion averaging used by PCL 1.12.
        for (const auto& vote : cluster) {
            translation_average += vote.pose.translation();
            rotation_average += Eigen::Quaternionf(vote.pose.rotation()).coeffs();
        }
        translation_average /= static_cast<float>(cluster.size());
        rotation_average /= static_cast<float>(cluster.size());

        if (rotation_average.norm() <= 1e-12f) {
            continue;
        }

        CompatPoseWithVotes out;
        out.pose = Eigen::Affine3f::Identity();
        out.pose.translation() = translation_average;
        out.pose.linear() =
            Eigen::Quaternionf(rotation_average).normalized().toRotationMatrix();
        out.votes = cluster_votes[cid];
        result.push_back(out);
    }
    return result;
}

static std::vector<CompatPoseWithVotes> compat_ppf_candidates_pcl112(
    const pcl::PointCloud<pcl::PointNormal>::ConstPtr& model,
    const pcl::PointCloud<pcl::PointNormal>::ConstPtr& scene,
    const pcl::PPFHashMapSearch::Ptr& search,
    int scene_reference_rate,
    float position_cluster_threshold,
    float rotation_cluster_threshold_rad,
    int max_candidates
) {
    if (!search) {
        throw std::runtime_error("PPF search method is null");
    }
    if (model->empty() || scene->empty()) {
        throw std::runtime_error("PPF model/scene must not be empty");
    }

    auto scene_tree = pcl::make_shared<pcl::KdTreeFLANN<pcl::PointNormal>>();
    scene_tree->setInputCloud(scene);

    const std::size_t angle_bins = static_cast<std::size_t>(
        std::floor(2.0 * M_PI / search->getAngleDiscretizationStep())
    );
    if (angle_bins == 0) {
        throw std::runtime_error("Invalid PPF angle discretization");
    }

    std::vector<std::vector<unsigned int>> accumulator(
        model->size(),
        std::vector<unsigned int>(angle_bins, 0)
    );
    std::vector<CompatPoseWithVotes> voted_poses;

    float f1 = 0.f, f2 = 0.f, f3 = 0.f, f4 = 0.f;

    // PCL 1.12: every N-th scene point is a reference point.
    for (pcl::index_t sr = 0;
         sr < static_cast<pcl::index_t>(scene->size());
         sr += scene_reference_rate) {

        const Eigen::Vector3f scene_ref_point = (*scene)[sr].getVector3fMap();
        const Eigen::Vector3f scene_ref_normal = (*scene)[sr].getNormalVector3fMap();

        const float dot_s = std::max(
            -1.0f, std::min(1.0f, scene_ref_normal.dot(Eigen::Vector3f::UnitX()))
        );
        const float rotation_angle_sg = std::acos(dot_s);
        const bool parallel_sg =
            scene_ref_normal.y() == 0.0f && scene_ref_normal.z() == 0.0f;
        const Eigen::Vector3f rotation_axis_sg =
            parallel_sg
                ? Eigen::Vector3f::UnitY()
                : scene_ref_normal.cross(Eigen::Vector3f::UnitX()).normalized();
        const Eigen::AngleAxisf rotation_sg(rotation_angle_sg, rotation_axis_sg);
        const Eigen::Affine3f transform_sg(
            Eigen::Translation3f(rotation_sg * (-scene_ref_point)) * rotation_sg
        );

        pcl::Indices indices;
        std::vector<float> distances;
        scene_tree->radiusSearch(
            (*scene)[sr],
            search->getModelDiameter() / 2.0f,
            indices,
            distances
        );

        for (const auto scene_index : indices) {
            if (scene_index == sr) continue;

            if (!pcl::computePairFeatures(
                    (*scene)[sr].getVector4fMap(),
                    (*scene)[sr].getNormalVector4fMap(),
                    (*scene)[scene_index].getVector4fMap(),
                    (*scene)[scene_index].getNormalVector4fMap(),
                    f1, f2, f3, f4)) {
                continue;
            }

            std::vector<std::pair<std::size_t, std::size_t>> nearest_indices;
            search->nearestNeighborSearch(f1, f2, f3, f4, nearest_indices);

            const Eigen::Vector3f scene_point =
                (*scene)[scene_index].getVector3fMap();
            const Eigen::Vector3f scene_point_transformed =
                transform_sg * scene_point;

            float alpha_s = std::atan2(
                -scene_point_transformed.z(),
                scene_point_transformed.y()
            );
            if (std::sin(alpha_s) * scene_point_transformed.z() < 0.0f) {
                alpha_s *= -1.0f;
            }
            alpha_s *= -1.0f;

            for (const auto& nearest_index : nearest_indices) {
                const std::size_t model_ref = nearest_index.first;
                const std::size_t model_point = nearest_index.second;
                if (model_ref >= accumulator.size()) continue;
                if (model_ref >= search->alpha_m_.size() ||
                    model_point >= search->alpha_m_[model_ref].size()) {
                    continue;
                }

                float alpha = search->alpha_m_[model_ref][model_point] - alpha_s;
                if (alpha < -M_PI) alpha += static_cast<float>(2.0 * M_PI);
                else if (alpha > M_PI) alpha -= static_cast<float>(2.0 * M_PI);

                std::size_t alpha_bin = static_cast<std::size_t>(
                    std::floor(
                        (alpha + static_cast<float>(M_PI)) /
                        search->getAngleDiscretizationStep()
                    )
                );
                if (alpha_bin >= angle_bins) alpha_bin = angle_bins - 1;
                accumulator[model_ref][alpha_bin] += 1;
            }
        }

        // PCL 1.12 takes the single highest peak for each scene reference.
        std::size_t max_i = 0;
        std::size_t max_j = 0;
        unsigned int max_votes = 0;

        for (std::size_t i = 0; i < accumulator.size(); ++i) {
            for (std::size_t j = 0; j < accumulator[i].size(); ++j) {
                if (accumulator[i][j] > max_votes) {
                    max_votes = accumulator[i][j];
                    max_i = i;
                    max_j = j;
                }
                accumulator[i][j] = 0;
            }
        }

        if (max_votes == 0 || max_i >= model->size()) {
            continue;
        }

        const Eigen::Vector3f model_ref_point = (*model)[max_i].getVector3fMap();
        const Eigen::Vector3f model_ref_normal = (*model)[max_i].getNormalVector3fMap();

        const float dot_m = std::max(
            -1.0f, std::min(1.0f, model_ref_normal.dot(Eigen::Vector3f::UnitX()))
        );
        const float rotation_angle_mg = std::acos(dot_m);
        const bool parallel_mg =
            model_ref_normal.y() == 0.0f && model_ref_normal.z() == 0.0f;
        const Eigen::Vector3f rotation_axis_mg =
            parallel_mg
                ? Eigen::Vector3f::UnitY()
                : model_ref_normal.cross(Eigen::Vector3f::UnitX()).normalized();
        const Eigen::AngleAxisf rotation_mg(rotation_angle_mg, rotation_axis_mg);
        const Eigen::Affine3f transform_mg(
            Eigen::Translation3f(rotation_mg * (-model_ref_point)) * rotation_mg
        );

        const Eigen::Affine3f pose =
            transform_sg.inverse() *
            Eigen::AngleAxisf(
                (static_cast<float>(max_j) + 0.5f) *
                    search->getAngleDiscretizationStep() -
                    static_cast<float>(M_PI),
                Eigen::Vector3f::UnitX()
            ) *
            transform_mg;

        voted_poses.push_back({pose, max_votes});
    }

    return compat_cluster_poses_pcl112(
        std::move(voted_poses),
        position_cluster_threshold,
        rotation_cluster_threshold_rad,
        max_candidates
    );
}

static py::dict ppf_register_py(
    const ArrayF& model_points,
    const ArrayF& scene_points,
    py::object model_normals_obj,
    py::object scene_normals_obj,
    float normal_radius,
    py::object model_viewpoint,
    py::object scene_viewpoint,
    float angle_step_deg,
    float distance_step,
    int scene_reference_rate,
    float position_cluster_threshold,
    float rotation_cluster_threshold_deg,
    int max_candidates,
    int threads
) {
    require_positive(normal_radius, "normal_radius");
    require_positive(angle_step_deg, "angle_step_deg");
    require_positive(distance_step, "distance_step");
    require_positive(position_cluster_threshold, "position_cluster_threshold");
    require_positive(rotation_cluster_threshold_deg, "rotation_cluster_threshold_deg");
    if (scene_reference_rate < 1) {
        throw std::invalid_argument("scene_reference_rate must be >= 1");
    }
    if (max_candidates < 1) {
        throw std::invalid_argument("max_candidates must be >= 1");
    }

    auto model_xyz = xyz_cloud_from_numpy(model_points);
    auto scene_xyz = xyz_cloud_from_numpy(scene_points);
    auto model_n = normals_or_estimate(
        model_xyz, model_normals_obj, normal_radius, model_viewpoint, threads
    );
    auto scene_n = normals_or_estimate(
        scene_xyz, scene_normals_obj, normal_radius, scene_viewpoint, threads
    );

    auto model = combine_xyz_normals(model_xyz, model_n);
    auto scene = combine_xyz_normals(scene_xyz, scene_n);

    auto model_features = pcl::make_shared<pcl::PointCloud<pcl::PPFSignature>>();
    pcl::PPFEstimation<pcl::PointNormal, pcl::PointNormal, pcl::PPFSignature> ppf;
    ppf.setInputCloud(model);
    ppf.setInputNormals(model);
    ppf.compute(*model_features);

    const float angle_step_rad =
        angle_step_deg * static_cast<float>(M_PI) / 180.0f;
    auto search = pcl::make_shared<pcl::PPFHashMapSearch>(
        angle_step_rad,
        distance_step
    );
    search->setInputFeatureCloud(model_features);

    std::vector<CompatPoseWithVotes> compat_candidates;
    Eigen::Matrix4f final_m = Eigen::Matrix4f::Identity();
    bool converged = false;
    std::string candidate_backend;

#if PCL_VERSION_COMPARE(>=, 1, 14, 0)
    // PCL >= 1.14 exposes the clustered candidate list directly.
    pcl::PPFRegistration<pcl::PointNormal, pcl::PointNormal> reg;
    reg.setInputSource(model);
    reg.setInputTarget(scene);
    reg.setSearchMethod(search);
    reg.setSceneReferencePointSamplingRate(
        static_cast<unsigned int>(scene_reference_rate)
    );
    reg.setPositionClusteringThreshold(position_cluster_threshold);
    reg.setRotationClusteringThreshold(
        rotation_cluster_threshold_deg * static_cast<float>(M_PI) / 180.0f
    );

    pcl::PointCloud<pcl::PointNormal> aligned;
    reg.align(aligned);

    const auto pcl_candidates = reg.getBestPoseCandidates();
    const std::size_t k = std::min<std::size_t>(
        pcl_candidates.size(),
        static_cast<std::size_t>(max_candidates)
    );
    compat_candidates.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        CompatPoseWithVotes x;
        x.pose = pcl_candidates[i].pose;
        x.votes = pcl_candidates[i].votes;
        compat_candidates.push_back(x);
    }

    final_m = reg.getFinalTransformation();
    converged = reg.hasConverged();
    candidate_backend = "PCL::PPFRegistration::getBestPoseCandidates";
#else
    // Ubuntu 22.04 ships PCL 1.12. Its PPFRegistration has no public
    // getBestPoseCandidates(). This compatibility path mirrors the PCL 1.12
    // voting/clustering algorithm using PCL's PPFHashMapSearch and pair features,
    // while retaining the ranked clusters for Python.
    compat_candidates = compat_ppf_candidates_pcl112(
        model,
        scene,
        search,
        scene_reference_rate,
        position_cluster_threshold,
        rotation_cluster_threshold_deg * static_cast<float>(M_PI) / 180.0f,
        max_candidates
    );
    if (!compat_candidates.empty()) {
        final_m = compat_candidates.front().pose.matrix();
        converged = true;
    }
    candidate_backend = "PCL-1.12-compatible candidate backport";
#endif

    const std::size_t k = compat_candidates.size();
    auto transforms = make_array<float>({
        static_cast<py::ssize_t>(k), 4, 4
    });
    auto votes = make_array<std::uint32_t>({
        static_cast<py::ssize_t>(k)
    });
    auto T = transforms.mutable_unchecked<3>();
    auto V = votes.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(k); ++i) {
        const auto M = compat_candidates[static_cast<std::size_t>(i)].pose.matrix();
        V(i) = compat_candidates[static_cast<std::size_t>(i)].votes;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                T(i,r,c) = M(r,c);
            }
        }
    }

    auto final_transform = make_array<float>({4, 4});
    auto F = final_transform.mutable_unchecked<2>();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            F(r,c) = final_m(r,c);
        }
    }

    py::dict d;
    d["converged"] = converged;
    d["transform"] = final_transform;
    d["transforms"] = transforms;
    d["votes"] = votes;
    d["model_feature_count"] = static_cast<py::int_>(model_features->size());
    d["model_point_count"] = static_cast<py::int_>(model->size());
    d["scene_point_count"] = static_cast<py::int_>(scene->size());
    d["angle_step_deg"] = angle_step_deg;
    d["distance_step"] = distance_step;
    d["scene_reference_rate"] = scene_reference_rate;
    d["candidate_backend"] = candidate_backend;
    return d;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Small pybind11 bridge to selected PCL 3D algorithms";

    m.def("voxel_downsample", &voxel_downsample_py,
          py::arg("points"), py::arg("leaf_size"));

    m.def("estimate_normals", &estimate_normals_py,
          py::arg("points"), py::arg("radius"),
          py::arg("viewpoint") = py::none(),
          py::arg("threads") = 0);

    m.def("harris3d", &harris3d_py,
          py::arg("points"), py::arg("radius"),
          py::arg("threshold") = 0.0f,
          py::arg("nonmax") = true,
          py::arg("refine") = false,
          py::arg("method") = "HARRIS",
          py::arg("normals") = py::none(),
          py::arg("threads") = 0);

    m.def("iss3d", &iss3d_py,
          py::arg("points"),
          py::arg("salient_radius"),
          py::arg("nonmax_radius"),
          py::arg("gamma21") = 0.975f,
          py::arg("gamma32") = 0.975f,
          py::arg("min_neighbors") = 5,
          py::arg("threads") = 0);

    m.def("sift3d", &sift3d_py,
          py::arg("points"), py::arg("intensity"),
          py::arg("min_scale"),
          py::arg("n_octaves") = 3,
          py::arg("n_scales_per_octave") = 4,
          py::arg("min_contrast") = 0.001f);

    m.def("fpfh", &fpfh_py,
          py::arg("points"), py::arg("radius"),
          py::arg("normals") = py::none(),
          py::arg("normal_radius") = 0.005f,
          py::arg("viewpoint") = py::none(),
          py::arg("threads") = 0);

    m.def("shot", &shot_py,
          py::arg("points"), py::arg("radius"),
          py::arg("normals") = py::none(),
          py::arg("normal_radius") = 0.005f,
          py::arg("viewpoint") = py::none());

    m.def("ppf_register", &ppf_register_py,
          py::arg("model_points"),
          py::arg("scene_points"),
          py::arg("model_normals") = py::none(),
          py::arg("scene_normals") = py::none(),
          py::arg("normal_radius") = 0.005f,
          py::arg("model_viewpoint") = py::none(),
          py::arg("scene_viewpoint") = py::none(),
          py::arg("angle_step_deg") = 12.0f,
          py::arg("distance_step") = 0.01f,
          py::arg("scene_reference_rate") = 5,
          py::arg("position_cluster_threshold") = 0.01f,
          py::arg("rotation_cluster_threshold_deg") = 20.0f,
          py::arg("max_candidates") = 20,
          py::arg("threads") = 0);
}
