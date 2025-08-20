#ifndef MSE_H
#define MSE_H

#include <vector>
#include <cmath>
#include <stdexcept>

// Membuat Fungsi untuk menghitung Mean Squared Error (MSE) //
class MSE {
public:
    // Fungsi untuk menghitung MSE //
    static double Calculate(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            throw std::invalid_argument("Ukuran vektor y_true dan y_pred harus sama dan tidak boleh kosong!");
        }
        
        // Rumus MSE: MSE = (1/n) * Σ (y_true – y_pred)^2. //
        double sum_squared_error = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            double error = y_true[i] - y_pred[i];
            sum_squared_error += error * error; // Menghitung selisih kuadrat //
        }

        return sum_squared_error / y_true.size(); // Mengembalikan rata-rata selisih kuadrat //
    }
};

#endif
