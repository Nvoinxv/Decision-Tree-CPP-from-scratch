// Persiapkan Libary dan header yang di butuhkan //
// Ini udah gw setup headernya di include //
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono> 
#include <random>
#include "../include/LabelingEncoder.h"
#include "../include/Read_csv.h"
#include "../include/DecisonTree.h"
#include "../include/MSE.h"
#include "../include/HapusOutlier.h"
#include "../include/R2_SCORE.h"
#include "../include/MinMaxScaling.h"
#include "../include/OptimizerAdam.h"
#include "../include/RegulasiL2.h"

// Main utama program //
int main() {
    // Baca dataset dari file CSV //
    std::vector<std::vector<std::string>> Dataset = readCSVString("D:\\DecisionTreeC++\\data\\insurance.csv", true);
    
    // Cek apakah dataset berhasil dibaca //
    if (Dataset.empty()) {
        std::cerr << "Error: Dataset kosong atau gagal dibaca!" << std::endl;
        return -1;
    }
    
    std::cout << "Dataset berhasil dibaca dengan " << Dataset.size() << " baris dan " 
         << Dataset[0].size() << " kolom" << std::endl;
    
    // Pisahkan fitur (X) dan target (y) //
    // Asumsi kolom terakhir adalah target variable //
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    
    for (const auto& row : Dataset) {
        std::vector<double> features;
        for (size_t i = 0; i < row.size() - 1; ++i) {
            try {
                features.push_back(std::stod(row[i])); // konversi string ke double //
            } catch (const std::invalid_argument&) {
                features.push_back(0.0); // kalau gagal konversi jadi 0.0 //
            }
        }
        X.push_back(features);

        try {
            y.push_back(std::stod(row[row.size() - 1])); // kolom terakhir sebagai target numerik //
        } catch (const std::invalid_argument&) {
            y.push_back(0.0);
        }
    }
    
    std::cout << "Data berhasil dipisah: " << X.size() << " samples dengan " 
         << X[0].size() << " fitur" << std::endl;
    
    // Labeling data data string jadi numerik agar perhitungan nya makin bagus //
    std::cout << "\n=== Labeling Data Jadi Numerik ===" << std::endl;
    LabelEncoder sexEncoder;
    LabelEncoder smokerEncoder;
    LabelEncoder regionEncoder;
    
    std::vector<std::string> sex_col, smoker_col, region_col;
    for (const auto& row : Dataset) {
        sex_col.push_back(row[1]);    // kolom sex //
        smoker_col.push_back(row[4]); // kolom smoker //
    }

    sexEncoder.fit(sex_col);
    smokerEncoder.fit(smoker_col);

    // Clear X dan y sebelum diisi ulang hasil encoding //
    X.clear();
    y.clear();

    // Transform data mentah menjadi numerik (fitur X dan target y) //
    for (const auto& row : Dataset) {
        std::vector<double> features;

        // Kolom 0: age (numerik langsung masuk) //
        features.push_back(std::stod(row[0]));

        // Kolom 1: sex (string -> encode) //
        features.push_back(sexEncoder.transform({row[1]})[0]);

        // Kolom 2: bmi (numerik) //
        features.push_back(std::stod(row[2]));

        // Kolom 3: children (numerik) //
        features.push_back(std::stod(row[3]));

        // Kolom 4: smoker (string -> encode) //
        features.push_back(smokerEncoder.transform({row[4]})[0]);

        X.push_back(features);

        // Kolom 5: charges (target numerik) //
        y.push_back(std::stod(row[6]));
    }
    
    std::cout << "Data berhasil dipisah: " << X.size() << " samples dengan " 
         << X[0].size() << " fitur" << std::endl;
    
    std::cout << "\nContoh 5 Sample hasil labeling: " << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, X.size()); ++i) {
            for (size_t j = 0; j < X[i].size(); ++j) {
                std::cout << X[i][j] << " ";
            }
        std::cout << std::endl;
    }

    // Shuffle Data biar random dan hasil prediksi nya bisa di perhitungkan //
    std::cout << "\n==== Shuffle Data ====" << std::endl;
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Apply ke variabel X dan Y //
    // Atau kita sebut, mengubah data nya jadi acak di features dan target //
    std::vector<std::vector<double>> X_shuffled;
    std::vector<double> y_shuffled;
    for (size_t idx : indices) {
        X_shuffled.push_back(X[idx]);
        y_shuffled.push_back(y[idx]);
    }
    X = X_shuffled;
    y = y_shuffled;
    std::cout << "Data berhasil di-shuffle" << std::endl;
    
    // Labeling data data string jadi numerik agar perhitungan nya makin bagus //
    std::cout << "\n=== Labeling Data Jadi Numerik ===" << std::endl;
    std::cout << "Contoh hasil encode sex (0/1): " << sex_col[0] << " -> " 
              << sexEncoder.transform({sex_col[0]})[0] << std::endl;
    std::cout << "Contoh hasil encode smoker (0/1): " << smoker_col[0] << " -> " 
              << smokerEncoder.transform({smoker_col[0]})[0] << std::endl;

    // Hapus outlier dari target variable menggunakan quantil //
    std::cout << "\n=== Preprocessing: Hapus Outlier ===" << std::endl;
    Quantil outlierDetector;
    outlierDetector.fit(y);
    
    std::cout << "Q1: " << outlierDetector.getQ1() << std::endl;
    std::cout << "Q3: " << outlierDetector.getQ3() << std::endl;
    std::cout << "IQR: " << outlierDetector.getIQR() << std::endl;
    std::cout << "Lower Bound: " << outlierDetector.getLowerBound() << std::endl;
    std::cout << "Upper Bound: " << outlierDetector.getUpperBound() << std::endl;
    
    // Filter data berdasarkan outlier bounds //
    std::vector<std::vector<double>> X_clean;
    std::vector<double> y_clean;
    
    for (size_t i = 0; i < y.size(); ++i) {
        if (y[i] >= outlierDetector.getLowerBound() && y[i] <= outlierDetector.getUpperBound()) {
            X_clean.push_back(X[i]);
            y_clean.push_back(y[i]);
        }
    }
    
    std::cout << "Data setelah hapus outlier: " << X_clean.size() << " samples" << std::endl;
    
    // Normalisasi data menggunakan MinMaxScaler //
    std::cout << "\n=== Preprocessing: MinMax Scaling ===" << std::endl;
    MinMaxScaler scaler;
    scaler.fit_transform(X_clean);
    std::cout << "Data berhasil dinormalisasi ke range [0,1]" << std::endl;
    
    std::cout << "\nContoh 5 sample hasil scaling:" << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, X_clean.size()); ++i) {
            for (size_t j = 0; j < X_clean[i].size(); ++j) {
                std::cout << X_clean[i][j] << " ";
            }
        std::cout << std::endl;
    }

    // Split data untuk training dan testing (80:20) //
    std::cout << "\n=== Split Data Training dan Testing ===" << std::endl;
    size_t train_size = (size_t)(X_clean.size() * 0.8);
    
    std::vector<std::vector<double>> X_train(X_clean.begin(), X_clean.begin() + train_size);
    std::vector<double> y_train(y_clean.begin(), y_clean.begin() + train_size);
    
    std::vector<std::vector<double>> X_test(X_clean.begin() + train_size, X_clean.end());
    std::vector<double> y_test(y_clean.begin() + train_size, y_clean.end());
    
    std::cout << "Data training: " << X_train.size() << " samples" << std::endl;
    std::cout << "Data testing: " << X_test.size() << " samples" << std::endl;
    
    // Inisialisasi Decision Tree Regressor //
    std::cout << "\n=== Training Decision Tree ===" << std::endl;
    DecisionTreeRegressor dt(8, 3); // max_depth=5, min_samples_split=2 //
    
    // Mulai training dengan mengukur waktu //
    auto start_time = std::chrono::high_resolution_clock::now();
    dt.fit(X_train, y_train);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Training selesai dalam " << duration.count() << " ms" << std::endl;

    // Backward pass untuk improvement model //
    std::cout << "\n=== Improvement Model dengan Backward Pass ===" << std::endl;
    std::cout << "Melakukan backward pass untuk improve model..." << std::endl;
    
    // Update model dengan residual training //
    dt.backward(X_train, y_train);
    
    // Evaluasi ulang setelah backward //
    std::vector<double> y_train_improved;
    std::vector<double> y_test_improved;
    
    for (const auto& sample : X_train) {
        y_train_improved.push_back(dt.forward(sample));
    }
    
    for (const auto& sample : X_test) {
        y_test_improved.push_back(dt.forward(sample));
    }
    
    double mse = MSE::Calculate(y_train, y_train_improved);
    double r2_score = R2_Score::CalculateR2Score(y_train, y_train_improved);

    std::cout << "MSE: " << mse << std::endl;
    std::cout << "R2_SCORE: " << r2_score << std::endl;

    // Contoh prediksi single sample //
    std::cout << "\n=== Contoh Prediksi Single Sample ===" << std::endl;
    if (!X_test.empty()) {
        std::vector<double> sample = X_test[0];
        double prediction = dt.forward(sample);
        double actual = y_test[0];
        
        std::cout << "Sample pertama dari test set:" << std::endl;
        std::cout << "Fitur: [";
        for (size_t i = 0; i < sample.size(); ++i) {
            std::cout << sample[i];
            if (i < sample.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Prediksi: " << prediction << std::endl;
        std::cout << "Actual: " << actual << std::endl;
        std::cout << "Error: " << abs(actual - prediction) << std::endl;
    }
    
    std::cout << "\n=== Program Selesai ===" << std::endl;
    
    return 0;
}