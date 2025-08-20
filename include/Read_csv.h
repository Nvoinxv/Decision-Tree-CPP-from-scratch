#ifndef READ_CSV_H
#define READ_CSV_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Membuat Fungsi untuk membaca file CSV numerik //
// Output berupa vector 2D bertipe double //
std::vector<std::vector<double>> readCSV(const std::string& filename, bool skipHeader = false, int maxRows = -1) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: tidak bisa membuka file " << filename << std::endl;
        return data;
    }

    std::string line;
    if (skipHeader) std::getline(file, line); // skip header kalau ada //

    int rowCount = 0;
    while (std::getline(file, line)) {
        if (maxRows > 0 && rowCount >= maxRows) break;

        std::vector<double> row;
        size_t pos = 0;
        int col = 0;

        while ((pos = line.find(',')) != std::string::npos) {
            std::string token = line.substr(0, pos);
            if (col > 0) { // skip kolom tipe data waktu //
                try {
                    row.push_back(std::stod(token));
                } catch (const std::invalid_argument&) {
                    row.push_back(0.0); // kalau gagal konversi jadi 0.0 //
                }
            }
            line.erase(0, pos + 1);
            col++;
        }
        // tambahin kolom terakhir //
        try {
            row.push_back(std::stod(line));
        } catch (const std::invalid_argument&) {
            row.push_back(0.0);
        }

        data.push_back(row);
        rowCount++;
    }

    file.close();
    return data;
}

// Fungsi baca CSV kolom tunggal numerik //
// Output berupa vector bertipe double //
std::vector<double> readCSVColumn(const std::string& filename, int colIndex = 0, bool skipHeader = false, int maxRows = -1) {
    std::vector<double> column;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: tidak bisa membuka file " << filename << std::endl;
        return column;
    }

    std::string line;
    if (skipHeader) std::getline(file, line);

    int rowCount = 0;
    while (std::getline(file, line)) {
        if (maxRows > 0 && rowCount >= maxRows) break;

        std::stringstream ss(line);
        std::string cell;
        int idx = 0;
        while (std::getline(ss, cell, ',')) {
            if (idx == colIndex) {
                try {
                    column.push_back(std::stod(cell));
                } catch (const std::invalid_argument&) {
                    std::cerr << "Error konversi: " << cell << " bukan angka!" << std::endl;
                    column.push_back(0.0);
                }
                break;
            }
            idx++;
        }

        rowCount++;
    }

    file.close();
    return column;
}

// Membuat Fungsi untuk membaca file CSV bertipe string //
// Output berupa vector 2D bertipe string //
// Berguna untuk data kategorikal yang tidak bisa langsung jadi angka //
std::vector<std::vector<std::string>> readCSVString(const std::string& filename, bool skipHeader = false, int maxRows = -1) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: tidak bisa membuka file " << filename << std::endl;
        return data;
    }

    std::string line;
    if (skipHeader) std::getline(file, line); // skip header kalau ada //

    int rowCount = 0;
    while (std::getline(file, line)) {
        if (maxRows > 0 && rowCount >= maxRows) break;

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(cell); // simpan langsung sebagai string //
        }

        data.push_back(row);
        rowCount++;
    }

    file.close();
    return data;
}

#endif
