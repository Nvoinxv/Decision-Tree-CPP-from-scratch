#ifndef R2_SCORE_H
#define R2_SCORE_H

#include <cmath>
#include <vector>
#include <stdexcept>
#include <numeric>

// Membuat Fungsi untuk menghitung R² Score //
class R2_Score {
public:
    // Fungsi untuk menghitung R² //
    static double CalculateR2Score(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Ukuran vektor y_true dan y_pred harus sama dan tidak boleh kosong!");
        }

        // Hitung mean dari y_true //
        double mean_y_true = std::accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();

        // Hitung SSres dan SStot //
        double ss_res = 0.0; // Residual Sum of Squares //
        double ss_tot = 0.0; // Total Sum of Squares //

        for (size_t i = 0; i < y_true.size(); ++i) {
            double residual = y_true[i] - y_pred[i];
            ss_res += residual * residual;

            double diff_mean = y_true[i] - mean_y_true;
            ss_tot += diff_mean * diff_mean;
        }

        // Rumus R² = 1 - (SSres / SStot) //
        return 1.0 - (ss_res / ss_tot);
    }
};

#endif
