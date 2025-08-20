#ifndef HAPUS_OUTLIER_H
#define HAPUS_OUTLIER_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

// Membuat perhitungan Outlier //
// Disini gw pakai perhitungan Quantil //
// Rumus nya: Q3 - Q1 //
// Rumus 2 nya: lower_bound = Q1 - 1.5 * IQR //
// Rumus 3 nya: upper_bound = Q3 + 1.5 * IQR //

class Quantil {
private: 
    double Q1; // Kuartil 1 (25%) //
    double Q3; // Kuartil 3 (75%) //
    double IQR; // Interquartile Range //
    double lower_bound; // Batas bawah outlier //
    double upper_bound; // Batas atas outlier //

    // Fungsi bantu untuk mencari quantil //
    double getQuantile(std::vector<double> data, double q) {
        std::sort(data.begin(), data.end());
        double pos = (data.size() - 1) * q;
        int idx = std::floor(pos);
        double frac = pos - idx;
        if (idx + 1 < data.size()) {
            return data[idx] * (1 - frac) + data[idx + 1] * frac;
        }
        return data[idx];
    }

public:
    // Constructor default //
    Quantil() : Q1(0), Q3(0), IQR(0), lower_bound(0), upper_bound(0) {}

    // Fungsi untuk menghitung Q1, Q3, dan IQR //
    void fit(const std::vector<double>& data) {
        if (data.empty()) {
            std::cerr << "Error: Data kosong, tidak bisa hitung quantil." << std::endl;
            return;
        }

        Q1 = getQuantile(data, 0.25); // Ambil 25% data //
        Q3 = getQuantile(data, 0.75); // Ambil 75% data //
        IQR = Q3 - Q1;                // Rumus: Q3 - Q1 //
        lower_bound = Q1 - 1.5 * IQR; // Rumus: Q1 - 1.5 * IQR //
        upper_bound = Q3 + 1.5 * IQR; // Rumus: Q3 + 1.5 * IQR //
    }

    // Fungsi untuk hapus outlier berdasarkan bound //
    std::vector<double> removeOutliers(const std::vector<double>& data) {
        std::vector<double> cleaned;
        for (double val : data) {
            if (val >= lower_bound && val <= upper_bound) {
                cleaned.push_back(val);
            }
        }
        return cleaned;
    }

    // Getter biar bisa ambil nilai Q1, Q3, IQR, bounds //
    double getQ1() const { return Q1; }
    double getQ3() const { return Q3; }
    double getIQR() const { return IQR; }
    double getLowerBound() const { return lower_bound; }
    double getUpperBound() const { return upper_bound; }
};

#endif
