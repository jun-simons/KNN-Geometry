#include <CGAL/Simple_cartesian.h>
#include <CGAL/Search_traits.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/K_neighbor_search.h>
#include <CGAL/Euclidean_distance.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Kernel = CGAL::Simple_cartesian<double>;
using FT = Kernel::FT;
using Point_d = std::vector<double>;

// --------------------
// Data structures
// --------------------
struct DataPoint
{
    Point_d features;
    double target;
};

struct NeighborInfo
{
    double target;
    double dist2;
};

struct QueryResult
{
    double prediction_uniform;
    double prediction_weighted;
    double confidence;
    std::vector<NeighborInfo> neighbors;
};

struct ExperimentSummary
{
    int k;
    double mse_uniform;
    double mse_weighted;
    double low25_error;
    double high25_error;
    double low50_error;
    double high50_error;
};

struct DatasetSplit
{
    std::vector<DataPoint> train;
    std::vector<DataPoint> test;
};

// --------------------
// CGAL traits
// --------------------
struct Construct_cartesian_const_iterator_d
{
    using result_type = Point_d::const_iterator;

    result_type operator()(const DataPoint &p) const
    {
        return p.features.begin();
    }

    result_type operator()(const DataPoint &p, int) const
    {
        return p.features.end();
    }
};

using Traits = CGAL::Search_traits<
    FT,
    DataPoint,
    Point_d::const_iterator,
    Construct_cartesian_const_iterator_d>;

using Distance = CGAL::Euclidean_distance<Traits>;
using Tree = CGAL::Kd_tree<Traits>;
using Neighbor_search = CGAL::K_neighbor_search<Traits, Distance>;

// --------------------
// Utility
// --------------------
std::vector<std::string> split_csv_line(const std::string &line)
{
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        tokens.push_back(item);
    }
    return tokens;
}

std::vector<DataPoint> load_csv_dataset(const std::string &filename, bool has_header = false)
{
    std::ifstream fin(filename);
    if (!fin)
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<DataPoint> data;
    std::string line;
    bool first_line = true;
    int expected_dim = -1;

    while (std::getline(fin, line))
    {
        if (line.empty())
            continue;

        if (first_line && has_header)
        {
            first_line = false;
            continue;
        }
        first_line = false;

        auto tokens = split_csv_line(line);
        if (tokens.size() < 2)
        {
            throw std::runtime_error("Each row must have at least 2 columns");
        }

        int dim = static_cast<int>(tokens.size()) - 1;
        if (expected_dim == -1)
        {
            expected_dim = dim;
        }
        else if (dim != expected_dim)
        {
            throw std::runtime_error("Inconsistent number of columns in CSV");
        }

        Point_d features;
        features.reserve(dim);
        for (int i = 0; i < dim; ++i)
        {
            features.push_back(std::stod(tokens[i]));
        }

        double target = std::stod(tokens.back());
        data.push_back({features, target});
    }

    if (data.empty())
    {
        throw std::runtime_error("Dataset is empty");
    }

    return data;
}

DatasetSplit train_test_split(
    std::vector<DataPoint> data,
    double train_fraction,
    unsigned seed = 42)
{
    if (train_fraction <= 0.0 || train_fraction >= 1.0)
    {
        throw std::invalid_argument("train_fraction must be in (0,1)");
    }

    std::mt19937 rng(seed);
    std::shuffle(data.begin(), data.end(), rng);

    size_t train_size = static_cast<size_t>(train_fraction * data.size());

    DatasetSplit split;
    split.train.assign(data.begin(), data.begin() + train_size);
    split.test.assign(data.begin() + train_size, data.end());
    return split;
}

double squared_distance(const Point_d &a, const Point_d &b)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error("Dimension mismatch in squared_distance");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// --------------------
// Prediction helpers
// --------------------
double predict_uniform_average(const std::vector<NeighborInfo> &neighbors)
{
    if (neighbors.empty())
    {
        throw std::runtime_error("No neighbors for prediction");
    }

    double sum = 0.0;
    for (const auto &n : neighbors)
    {
        sum += n.target;
    }
    return sum / neighbors.size();
}

double predict_weighted_average(const std::vector<NeighborInfo> &neighbors)
{
    if (neighbors.empty())
    {
        throw std::runtime_error("No neighbors for prediction");
    }

    double numerator = 0.0;
    double denominator = 0.0;

    for (const auto &n : neighbors)
    {
        double w = 1.0 / (std::sqrt(n.dist2) + 1e-9);
        numerator += w * n.target;
        denominator += w;
    }

    return numerator / denominator;
}

double compute_confidence(const std::vector<NeighborInfo> &neighbors)
{
    if (neighbors.empty())
        return 0.0;

    double sum = 0.0;
    for (const auto &n : neighbors)
    {
        sum += std::sqrt(n.dist2);
    }
    double avg_dist = sum / neighbors.size();
    return 1.0 / (1.0 + avg_dist);
}

// --------------------
// Brute force
// --------------------
QueryResult query_bruteforce(
    const std::vector<DataPoint> &train,
    const Point_d &query,
    int k)
{
    std::vector<std::pair<double, double>> dist_target;
    dist_target.reserve(train.size());

    for (const auto &p : train)
    {
        double dist2 = squared_distance(p.features, query);
        dist_target.push_back({dist2, p.target});
    }

    std::nth_element(
        dist_target.begin(),
        dist_target.begin() + k,
        dist_target.end(),
        [](const auto &a, const auto &b)
        {
            return a.first < b.first;
        });

    std::vector<NeighborInfo> neighbors;
    neighbors.reserve(k);

    for (int i = 0; i < k; ++i)
    {
        neighbors.push_back({dist_target[i].second, dist_target[i].first});
    }

    QueryResult result;
    result.prediction_uniform = predict_uniform_average(neighbors);
    result.prediction_weighted = predict_weighted_average(neighbors);
    result.confidence = compute_confidence(neighbors);
    result.neighbors = neighbors;
    return result;
}

// --------------------
// KD-tree
// --------------------
class KDTreeKNN
{
public:
    explicit KDTreeKNN(const std::vector<DataPoint> &train_data)
    {
        tree_ = std::make_unique<Tree>(train_data.begin(), train_data.end());
        tree_->build();
    }

    QueryResult query(const Point_d &query, int k) const
    {
        DataPoint q{query, 0.0};
        Neighbor_search search(*tree_, q, k, 0.0);

        std::vector<NeighborInfo> neighbors;
        neighbors.reserve(k);

        for (auto it = search.begin(); it != search.end(); ++it)
        {
            neighbors.push_back({it->first.target, it->second});
        }

        QueryResult result;
        result.prediction_uniform = predict_uniform_average(neighbors);
        result.prediction_weighted = predict_weighted_average(neighbors);
        result.confidence = compute_confidence(neighbors);
        result.neighbors = neighbors;
        return result;
    }

private:
    std::unique_ptr<Tree> tree_;
};

// --------------------
// Evaluation
// --------------------
double mse(
    const std::vector<double> &y_true,
    const std::vector<double> &y_pred)
{
    if (y_true.size() != y_pred.size())
    {
        throw std::runtime_error("mse: size mismatch");
    }

    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i)
    {
        double d = y_true[i] - y_pred[i];
        sum += d * d;
    }
    return sum / y_true.size();
}

double mean_abs_error_of_range(
    const std::vector<std::pair<double, double>> &conf_err,
    size_t begin_idx,
    size_t end_idx)
{
    if (begin_idx >= end_idx || end_idx > conf_err.size())
    {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = begin_idx; i < end_idx; ++i)
    {
        sum += conf_err[i].second;
    }
    return sum / static_cast<double>(end_idx - begin_idx);
}

// --------------------
// Main
// --------------------
int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: ./knn_kdtree_test <csv_file> <train_fraction> <k_list_comma_sep> [has_header]\n";
        std::cerr << "Example: ./knn_kdtree_test ../data/reg_20k_noisy.csv 0.8 1,3,5,7 0\n";
        return 1;
    }

    std::string filename = argv[1];
    double train_fraction = std::stod(argv[2]);
    std::string k_list_str = argv[3];
    bool has_header = (argc >= 5) ? (std::stoi(argv[4]) != 0) : false;

    try
    {
        auto data = load_csv_dataset(filename, has_header);
        auto split = train_test_split(data, train_fraction, 42);

        if (split.test.empty())
        {
            throw std::runtime_error("Test split is empty");
        }

        std::vector<int> k_values;
        {
            std::stringstream ss(k_list_str);
            std::string item;
            while (std::getline(ss, item, ','))
            {
                int k = std::stoi(item);
                if (k <= 0)
                {
                    throw std::runtime_error("All k values must be positive");
                }
                if (split.train.size() <= static_cast<size_t>(k))
                {
                    throw std::runtime_error("k too large for train set");
                }
                k_values.push_back(k);
            }
        }

        std::cout << "Loaded dataset: " << data.size() << " rows\n";
        std::cout << "Dimension: " << split.train[0].features.size() << "\n";
        std::cout << "Train size: " << split.train.size() << "\n";
        std::cout << "Test size: " << split.test.size() << "\n";
        std::cout << "k values: " << k_list_str << "\n\n";

        auto tree_build_start = std::chrono::high_resolution_clock::now();
        KDTreeKNN kd_model(split.train);
        auto tree_build_end = std::chrono::high_resolution_clock::now();
        double build_ms =
            std::chrono::duration<double, std::milli>(tree_build_end - tree_build_start).count();

        std::ofstream summary_csv("../data/summary_results.csv");
        summary_csv << "k,mse_uniform,mse_weighted,low25_error,high25_error,low50_error,high50_error\n";

        std::vector<ExperimentSummary> summaries;

        for (int k : k_values)
        {
            std::vector<double> y_true;
            std::vector<double> y_pred_uniform;
            std::vector<double> y_pred_weighted;
            std::vector<std::pair<double, double>> conf_abs_err;

            y_true.reserve(split.test.size());
            y_pred_uniform.reserve(split.test.size());
            y_pred_weighted.reserve(split.test.size());
            conf_abs_err.reserve(split.test.size());

            auto kd_query_start = std::chrono::high_resolution_clock::now();

            for (const auto &p : split.test)
            {
                QueryResult result = kd_model.query(p.features, k);

                y_true.push_back(p.target);
                y_pred_uniform.push_back(result.prediction_uniform);
                y_pred_weighted.push_back(result.prediction_weighted);

                double abs_err = std::abs(p.target - result.prediction_weighted);
                conf_abs_err.push_back({result.confidence, abs_err});
            }

            auto kd_query_end = std::chrono::high_resolution_clock::now();
            double query_ms =
                std::chrono::duration<double, std::milli>(kd_query_end - kd_query_start).count();

            std::sort(
                conf_abs_err.begin(),
                conf_abs_err.end(),
                [](const auto &a, const auto &b)
                {
                    return a.first < b.first;
                });

            size_t n = conf_abs_err.size();
            size_t q25 = n / 4;
            size_t q50 = n / 2;

            double low25 = mean_abs_error_of_range(conf_abs_err, 0, q25);
            double high25 = mean_abs_error_of_range(conf_abs_err, n - q25, n);
            double low50 = mean_abs_error_of_range(conf_abs_err, 0, q50);
            double high50 = mean_abs_error_of_range(conf_abs_err, q50, n);

            double mse_u = mse(y_true, y_pred_uniform);
            double mse_w = mse(y_true, y_pred_weighted);

            summaries.push_back({k, mse_u, mse_w, low25, high25, low50, high50});
            summary_csv << k << ","
                        << mse_u << ","
                        << mse_w << ","
                        << low25 << ","
                        << high25 << ","
                        << low50 << ","
                        << high50 << "\n";

            std::ofstream detail_csv("../data/details_k" + std::to_string(k) + ".csv");
            detail_csv << "confidence,abs_error\n";
            for (const auto &ce : conf_abs_err)
            {
                detail_csv << ce.first << "," << ce.second << "\n";
            }

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "k = " << k << "\n";
            std::cout << "  weighted MSE: " << mse_w << "\n";
            std::cout << "  uniform  MSE: " << mse_u << "\n";
            std::cout << "  low 25% confidence avg abs error:  " << low25 << "\n";
            std::cout << "  high 25% confidence avg abs error: " << high25 << "\n";
            std::cout << "  low 50% confidence avg abs error:  " << low50 << "\n";
            std::cout << "  high 50% confidence avg abs error: " << high50 << "\n";
            std::cout << "  query time (ms): " << query_ms << "\n\n";
        }

        std::cout << "KD-tree build time (ms): " << build_ms << "\n";
        std::cout << "Wrote summary_results.csv and details_k*.csv\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}