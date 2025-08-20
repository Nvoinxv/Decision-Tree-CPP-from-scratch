#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>

// Membuat Decision Tree untuk regresi //
// Sebenarnya ini adalah implementasi dasar dari Decision Tree //
// Cuma ini fleksibel berbagai pelatihan model kek regresi dan juga klasifikasi //
// Cuma disini gw buat regresi saja karna dataset nya emang lebih regresi //
class DecisionTreeRegressor {
private:
    struct Node {
        bool is_leaf;      // cek apakah node ini leaf //
        double value;      // nilai prediksi (kalau leaf) //
        int feature_index; // indeks fitur untuk split //
        double threshold;  // nilai threshold untuk split //
        Node* left;        // cabang kiri (ya) //
        Node* right;       // cabang kanan (tidak) //

        Node() : is_leaf(true), value(0.0), feature_index(-1), threshold(0.0), left(nullptr), right(nullptr) {}
    };

    Node* root;
    int max_depth;
    int min_samples_split;

public:
    DecisionTreeRegressor(int max_depth = 5, int min_samples_split = 2)
        : root(nullptr), max_depth(max_depth), min_samples_split(min_samples_split) {}

    ~DecisionTreeRegressor() {
        freeTree(root);
    }

    // Fungsi training tree //
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        if (X.empty() || y.empty() || X.size() != y.size()) {
            throw std::invalid_argument("Data input dan target harus memiliki ukuran yang sesuai dan tidak boleh kosong!");
        }
        root = buildTree(X, y, 0);
    }

    // Fungsi forward (prediksi) //
    double forward(const std::vector<double>& x) const {
        Node* node = root;
        while (node && !node->is_leaf) {
            if (x[node->feature_index] <= node->threshold) {
                node = node->left;
            } else {
                node = node->right;
            }
        }
        return node ? node->value : 0.0;
    }

    // Fungsi backward (pseudo gradient update) //
    // Catatan penting:
    // - Decision Tree tidak differentiable.
    // - Di sini TIDAK ada re-train / build ulang tree.
    // - Kita hanya lakukan koreksi bias global: geser semua leaf value
    //   dengan rata-rata residual (y_true - y_pred) * learning_rate kecil.
    void backward(const std::vector<std::vector<double>>& X, const std::vector<double>& y_true) {
        if (!root || X.empty() || y_true.empty() || X.size() != y_true.size()) return;

        // hitung prediksi sekarang //
        std::vector<double> y_pred;
        y_pred.reserve(X.size());
        for (const auto& row : X) {
            y_pred.push_back(forward(row));
        }

        // hitung residual (y_true - y_pred) //
        std::vector<double> residuals(y_true.size());
        for (size_t i = 0; i < y_true.size(); ++i) {
            residuals[i] = y_true[i] - y_pred[i];
        }

        // rata-rata residual sebagai bias correction //
        const double lr = 0.1; // learning rate kecil biar stabil //
        double bias = mean(residuals) * lr;

        // apply pergeseran bias ke semua leaf tanpa re-train //
        applyBiasShift(root, bias);
    }

private:
    // Membuat node baru dengan split terbaik //
    Node* buildTree(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int depth) {
        Node* node = new Node();

        // kalau mencapai depth max atau data terlalu sedikit, jadikan leaf //
        if (depth >= max_depth || y.size() < (size_t)min_samples_split) {
            node->is_leaf = true;
            node->value = mean(y); // leaf value = rata-rata target //
            return node;
        }

        // cari split terbaik //
        int best_feature = -1;
        double best_threshold = 0.0;
        double best_mse = std::numeric_limits<double>::infinity();

        for (size_t feature = 0; feature < X[0].size(); ++feature) {
            std::vector<double> feature_values;
            feature_values.reserve(X.size());
            for (const auto& row : X) {
                feature_values.push_back(row[feature]);
            }
            std::sort(feature_values.begin(), feature_values.end());

            for (size_t i = 1; i < feature_values.size(); ++i) {
                double threshold = (feature_values[i - 1] + feature_values[i]) / 2.0;

                // split data //
                std::vector<std::vector<double>> X_left, X_right;
                std::vector<double> y_left, y_right;
                X_left.reserve(X.size());
                X_right.reserve(X.size());
                y_left.reserve(y.size());
                y_right.reserve(y.size());

                for (size_t j = 0; j < X.size(); ++j) {
                    if (X[j][feature] <= threshold) {
                        X_left.push_back(X[j]);
                        y_left.push_back(y[j]);
                    } else {
                        X_right.push_back(X[j]);
                        y_right.push_back(y[j]);
                    }
                }

                if (y_left.empty() || y_right.empty()) continue;

                // hitung MSE untuk split ini //
                double mse_val = mse_split(y_left, y_right);
                if (mse_val < best_mse) {
                    best_mse = mse_val;
                    best_feature = static_cast<int>(feature);
                    best_threshold = threshold;
                }
            }
        }

        // kalau tidak ada split yang bagus, jadikan leaf //
        if (best_feature == -1) {
            node->is_leaf = true;
            node->value = mean(y);
            return node;
        }

        // bagi data sesuai split terbaik //
        std::vector<std::vector<double>> X_left, X_right;
        std::vector<double> y_left, y_right;
        X_left.reserve(X.size());
        X_right.reserve(X.size());
        y_left.reserve(y.size());
        y_right.reserve(y.size());

        for (size_t j = 0; j < X.size(); ++j) {
            if (X[j][best_feature] <= best_threshold) {
                X_left.push_back(X[j]);
                y_left.push_back(y[j]);
            } else {
                X_right.push_back(X[j]);
                y_right.push_back(y[j]);
            }
        }

        node->is_leaf = false;
        node->feature_index = best_feature;
        node->threshold = best_threshold;
        node->left = buildTree(X_left, y_left, depth + 1);
        node->right = buildTree(X_right, y_right, depth + 1);

        return node;
    }

    // Utility: apply bias shift ke semua leaf //
    void applyBiasShift(Node* node, double bias) {
        if (!node) return;
        if (node->is_leaf) {
            node->value += bias;
            return;
        }
        applyBiasShift(node->left, bias);
        applyBiasShift(node->right, bias);
    }

    // Karna gw pakai regresi jadi perhitungan evaluasi nya pakai MSE //
    // Sebenarnya sama aja mau klasifikasi dan regresi //
    // Yang membedakan hanya lah evaluasi saja //
    // Untuk klasifikasi pakai evaluasi Gini Index, Entropy, misclassification error //
    // Untuk regresi pakai MSE(Mean Squared Error), MAE(Mean Absolute Error), variance reduction //

    // Lalu bagian leaf nya itu kalau klasifikasi pilih kelas mayoritas dari data di leaf //
    // Untuk regresi sendiri leaf nya itu dari rata rata nilai target //
    // Hitung rata-rata //
    double mean(const std::vector<double>& y) const {
        return std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());
    }

    // Hitung MSE split //
    double mse_split(const std::vector<double>& y_left, const std::vector<double>& y_right) const {
        return mse(y_left) * y_left.size() + mse(y_right) * y_right.size();
    }

    // Hitung MSE satu sisi //
    double mse(const std::vector<double>& y) const {
        double mean_y = mean(y);
        double sum_sq = 0.0;
        for (double val : y) {
            double diff = val - mean_y;
            sum_sq += diff * diff;
        }
        return sum_sq / static_cast<double>(y.size());
    }

    // Hapus tree dari memori //
    void freeTree(Node* node) {
        if (node == nullptr) return;
        freeTree(node->left);
        freeTree(node->right);
        delete node;
    }
};

#endif
