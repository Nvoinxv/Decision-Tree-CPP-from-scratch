#ifndef MIN_MAX_SCALING_H
#define MIN_MAX_SCALING_H

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Membuat Class MinMaxScaler untuk preprocessing data //
// Jadi ini mengubah data ke rentang [0,1] //
// Untuk rumus MinMaxScaling adalah: (x - min(x)) / (max(x) - min(x)) //
// Data di normalisasi ini digunakan untuk mengurangi skala yang berlebihan pada data //

class MinMaxScaler {
private:
    std::vector<double> minValues;  // Menyimpan nilai minimum tiap kolom
    std::vector<double> maxValues;  // Menyimpan nilai maksimum tiap kolom
    bool fitted; // Flag untuk cek apakah scaler sudah fit

public:
    // Constructor //
    MinMaxScaler() : fitted(false) {}

    // Fungsi "fit" untuk mencari nilai min dan max dari setiap kolom //
    void fit(const std::vector<std::vector<double>>& data) {
        if (data.empty() || data[0].empty()) {
            std::cerr << "Error: Data tidak boleh kosong." << std::endl;
            return;
        }

        int n_cols = data[0].size();
        minValues.assign(n_cols, INFINITY);
        maxValues.assign(n_cols, -INFINITY);

        // Cari min dan max untuk setiap kolom //
        for (const auto& row : data) {
            for (int j = 0; j < n_cols; j++) {
                minValues[j] = std::min(minValues[j], row[j]);
                maxValues[j] = std::max(maxValues[j], row[j]);
            }
        }

        fitted = true;
    }

    // Fungsi "transform" untuk menormalisasi data ke range [0,1] //
    void transform(std::vector<std::vector<double>>& data) const {
        if (!fitted) {
            std::cerr << "Error: Scaler belum di-fit dengan data." << std::endl;
            return;
        }

        for (auto& row : data) {
            for (int j = 0; j < row.size(); j++) {
                if (maxValues[j] != minValues[j]) { 
                    row[j] = (row[j] - minValues[j]) / (maxValues[j] - minValues[j]);
                }
            }
        }
    }

    // Fungsi "fit_transform" untuk langsung fit + transform //
    void fit_transform(std::vector<std::vector<double>>& data) {
        fit(data);
        transform(data);
    }
};

#endif
