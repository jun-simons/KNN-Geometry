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
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

// allow any dimension
using Kernel = CGAL::Simple_cartesian<double>;
using FT = Kernel::FT;
using Point_d = std::vector<double>;

// --------------------
// Data structures
// --------------------
struct LabeledPoint
{
    Point_d features;
    int label;
    int id;
};

struct NeighborInfo
{
    int label;
    double dist2;
};

struct KNNResult
{
    int prediction_majority;
    int prediction_weighted;
    std::vector<NeighborInfo> neighbors;
};

// Cartesian iterator functor for LabeledPoint — only exposes features to CGAL
struct Construct_cartesian_const_iterator_d
{
    using result_type = Point_d::const_iterator;

    result_type operator()(const LabeledPoint &p) const
    {
        return p.features.begin();
    }

    result_type operator()(const LabeledPoint &p, int) const
    {
        return p.features.end();
    }
};

// Search traits use LabeledPoint as the tree's point type; label is ignored by CGAL
using Traits = CGAL::Search_traits<
    FT,
    LabeledPoint,
    Point_d::const_iterator,
    Construct_cartesian_const_iterator_d>;

using Distance = CGAL::Euclidean_distance<Traits>;
using Tree = CGAL::Kd_tree<Traits>;
using Neighbor_search = CGAL::K_neighbor_search<Traits, Distance>;

struct DatasetSplit
{
    std::vector<LabeledPoint> train;
    std::vector<LabeledPoint> test;
};

// -----------------------
// Utility functions
// ----------------------
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

DatasetSplit train_test_split(
    std::vector<LabeledPoint> data,
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

// load_csv_dataset
// Expected format
// -- accepts any number of columns
// -- last column is an integer label
// -- preceding columns are feature dimensions
std::vector<LabeledPoint> load_csv_dataset(const std::string &filename, bool has_header = false)
{
    std::ifstream fin(filename);
    if (!fin)
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<LabeledPoint> data;
    std::string line;
    bool first_line = true;
    int expected_dim = -1;
    int next_id = 0;

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
        // check consistency in cols and rows
        if (tokens.size() < 2)
        {
            throw std::runtime_error("Each row must have at least 2 columns (features + label)");
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

        int label = std::stoi(tokens.back());
        data.push_back({features, label, next_id++});
    }

    if (data.empty())
    {
        throw std::runtime_error("Dataset is empty");
    }

    return data;
}

// Squared Euclidean Distance
// note: could use other distance functions
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

int majority_vote(const std::vector<int> &labels)
{
    std::map<int, int> counts;
    for (int label : labels)
    {
        counts[label]++;
    }

    int best_label = -1;
    int best_count = -1;
    for (const auto &[label, count] : counts)
    {
        if (count > best_count)
        {
            best_count = count;
            best_label = label;
        }
    }
    return best_label;
}

int weighted_vote(const std::vector<NeighborInfo> &neighbors)
{
    std::map<int, double> weights;

    for (const auto &n : neighbors)
    {
        double w = 1.0 / (std::sqrt(n.dist2) + 1e-9);
        weights[n.label] += w;
    }

    int best_label = -1;
    double best_weight = -1.0;
    for (const auto &[label, weight] : weights)
    {
        if (weight > best_weight)
        {
            best_weight = weight;
            best_label = label;
        }
    }
    return best_label;
}

double average_neighbor_distance(const std::vector<NeighborInfo> &neighbors)
{
    if (neighbors.empty())
        return 0.0;

    double sum = 0.0;
    for (const auto &n : neighbors)
    {
        sum += std::sqrt(n.dist2);
    }
    return sum / neighbors.size();
}

double inverse_distance_confidence(const std::vector<NeighborInfo> &neighbors)
{
    double avg = average_neighbor_distance(neighbors);
    return 1.0 / (1.0 + avg);
}

// --- Brute-force KNN ---
// Linear search
KNNResult query_bruteforce(
    const std::vector<LabeledPoint> &train,
    const Point_d &query,
    int k)
{
    std::vector<std::pair<double, int>> dist_label;
    dist_label.reserve(train.size());

    // O(n) scan for all points
    for (const auto &p : train)
    {
        double dist2 = squared_distance(p.features, query);
        dist_label.push_back({dist2, p.label});
    }

    // take k closest points
    std::nth_element(
        dist_label.begin(),
        dist_label.begin() + k,
        dist_label.end(),
        [](const auto &a, const auto &b)
        {
            return a.first < b.first;
        });

    std::vector<int> labels;
    std::vector<NeighborInfo> neighbors;
    labels.reserve(k);
    neighbors.reserve(k);

    for (int i = 0; i < k; ++i)
    {
        labels.push_back(dist_label[i].second);
        neighbors.push_back({dist_label[i].second, dist_label[i].first});
    }

    KNNResult result;
    result.prediction_majority = majority_vote(labels);
    result.prediction_weighted = weighted_vote(neighbors);
    result.neighbors = neighbors;
    return result;
}

// --- KD-tree KNN wrapper ---
class KDTreeKNN
{
public:
    explicit KDTreeKNN(const std::vector<LabeledPoint> &train_data)
    {
        // Store LabeledPoint objects directly in the tree; CGAL uses only
        // the features iterator for spatial operations, label just "exists"
        tree_ = std::make_unique<Tree>(train_data.begin(), train_data.end());
        tree_->build();
    }

    KNNResult query(const Point_d &query, int k) const
    {
        LabeledPoint q{query, -1, -1};
        Neighbor_search search(*tree_, q, k, 0.0);

        std::vector<int> labels;
        std::vector<NeighborInfo> neighbors;
        labels.reserve(k);
        neighbors.reserve(k);

        for (auto it = search.begin(); it != search.end(); ++it)
        {
            labels.push_back(it->first.label);
            neighbors.push_back({it->first.label, it->second});
        }

        KNNResult result;
        result.prediction_majority = majority_vote(labels);
        result.prediction_weighted = weighted_vote(neighbors);
        result.neighbors = neighbors;
        return result;
    }

private:
    std::unique_ptr<Tree> tree_;
};

// Evaluation
double accuracy(
    const std::vector<int> &y_true,
    const std::vector<int> &y_pred)
{
    if (y_true.size() != y_pred.size())
    {
        throw std::runtime_error("accuracy: size mismatch");
    }

    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
    {
        if (y_true[i] == y_pred[i])
        {
            correct++;
        }
    }

    return static_cast<double>(correct) / y_true.size();
}

double average_of_vector(const std::vector<double> &values)
{
    if (values.empty())
        return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

// Main
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./knn_kdtree_test <csv_file> [k] [train_fraction] [has_header]\n";
        std::cerr << "Example: ./knn_kdtree_test data.csv 3 0.8 1\n";
        return 1;
    }

    std::string filename = argv[1];
    int k = (argc >= 3) ? std::stoi(argv[2]) : 3;
    double train_fraction = (argc >= 4) ? std::stod(argv[3]) : 0.8;
    bool has_header = (argc >= 5) ? (std::stoi(argv[4]) != 0) : false;

    try
    {
        auto data = load_csv_dataset(filename, has_header);

        if (data.size() <= static_cast<size_t>(k))
        {
            throw std::runtime_error("Dataset too small relative to k");
        }

        auto split = train_test_split(data, train_fraction, 42);

        if (split.train.size() <= static_cast<size_t>(k) || split.test.empty())
        {
            throw std::runtime_error("Train/test split invalid for chosen k");
        }

        const int dim = static_cast<int>(split.train[0].features.size());

        std::cout << "Loaded dataset: " << data.size() << " rows\n";
        std::cout << "Dimension: " << dim << "\n";
        std::cout << "Train size: " << split.train.size() << "\n";
        std::cout << "Test size: " << split.test.size() << "\n";
        std::cout << "k = " << k << "\n\n";

        // True labels
        std::vector<int> y_true;
        y_true.reserve(split.test.size());
        for (const auto &p : split.test)
        {
            y_true.push_back(p.label);
        }

        // Brute-force timing
        std::vector<int> brute_preds_majority;
        std::vector<int> brute_preds_weighted;
        std::vector<double> brute_conf_correct;
        std::vector<double> brute_conf_wrong;

        brute_preds_majority.reserve(split.test.size());
        brute_preds_weighted.reserve(split.test.size());

        auto brute_start = std::chrono::high_resolution_clock::now();
        for (const auto &q : split.test)
        {
            KNNResult result = query_bruteforce(split.train, q.features, k);
            brute_preds_majority.push_back(result.prediction_majority);
            brute_preds_weighted.push_back(result.prediction_weighted);

            double conf = inverse_distance_confidence(result.neighbors);
            if (result.prediction_majority == q.label)
            {
                brute_conf_correct.push_back(conf);
            }
            else
            {
                brute_conf_wrong.push_back(conf);
            }
        }
        auto brute_end = std::chrono::high_resolution_clock::now();

        double brute_ms =
            std::chrono::duration<double, std::milli>(brute_end - brute_start).count();

        double brute_acc_majority = accuracy(y_true, brute_preds_majority);
        double brute_acc_weighted = accuracy(y_true, brute_preds_weighted);

        // KD-tree timing
        auto tree_build_start = std::chrono::high_resolution_clock::now();
        KDTreeKNN kd_model(split.train);
        auto tree_build_end = std::chrono::high_resolution_clock::now();

        std::vector<int> kd_preds_majority;
        std::vector<int> kd_preds_weighted;
        std::vector<double> kd_conf_correct;
        std::vector<double> kd_conf_wrong;

        kd_preds_majority.reserve(split.test.size());
        kd_preds_weighted.reserve(split.test.size());

        auto kd_query_start = std::chrono::high_resolution_clock::now();
        for (const auto &q : split.test)
        {
            KNNResult result = kd_model.query(q.features, k);
            kd_preds_majority.push_back(result.prediction_majority);
            kd_preds_weighted.push_back(result.prediction_weighted);

            double conf = inverse_distance_confidence(result.neighbors);
            if (result.prediction_majority == q.label)
            {
                kd_conf_correct.push_back(conf);
            }
            else
            {
                kd_conf_wrong.push_back(conf);
            }
        }
        auto kd_query_end = std::chrono::high_resolution_clock::now();

        double build_ms =
            std::chrono::duration<double, std::milli>(tree_build_end - tree_build_start).count();

        double kd_query_ms =
            std::chrono::duration<double, std::milli>(kd_query_end - kd_query_start).count();

        double kd_acc_majority = accuracy(y_true, kd_preds_majority);
        double kd_acc_weighted = accuracy(y_true, kd_preds_weighted);

        // Output
        std::cout << std::fixed << std::setprecision(4);

        std::cout << "Brute-force majority-vote accuracy: " << brute_acc_majority << "\n";
        std::cout << "Brute-force weighted-vote accuracy: " << brute_acc_weighted << "\n";
        std::cout << "Brute-force total query time (ms): " << brute_ms << "\n";
        std::cout << "Brute-force avg/query (ms): " << (brute_ms / split.test.size()) << "\n";
        std::cout << "Brute-force avg confidence on correct predictions: "
                  << average_of_vector(brute_conf_correct) << "\n";
        std::cout << "Brute-force avg confidence on wrong predictions: "
                  << average_of_vector(brute_conf_wrong) << "\n\n";

        std::cout << "KD-tree majority-vote accuracy: " << kd_acc_majority << "\n";
        std::cout << "KD-tree weighted-vote accuracy: " << kd_acc_weighted << "\n";
        std::cout << "KD-tree build time (ms): " << build_ms << "\n";
        std::cout << "KD-tree total query time (ms): " << kd_query_ms << "\n";
        std::cout << "KD-tree avg/query (ms): " << (kd_query_ms / split.test.size()) << "\n";
        std::cout << "KD-tree avg confidence on correct predictions: "
                  << average_of_vector(kd_conf_correct) << "\n";
        std::cout << "KD-tree avg confidence on wrong predictions: "
                  << average_of_vector(kd_conf_wrong) << "\n\n";

        int mismatches_majority = 0;
        int mismatches_weighted = 0;

        for (size_t i = 0; i < brute_preds_majority.size(); ++i)
        {
            if (brute_preds_majority[i] != kd_preds_majority[i])
            {
                mismatches_majority++;
            }
            if (brute_preds_weighted[i] != kd_preds_weighted[i])
            {
                mismatches_weighted++;
            }
        }

        std::cout << "Prediction mismatches between methods (majority): "
                  << mismatches_majority << "\n";
        std::cout << "Prediction mismatches between methods (weighted): "
                  << mismatches_weighted << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}