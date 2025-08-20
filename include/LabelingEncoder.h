#ifndef LABEL_ENCODER_H
#define LABEL_ENCODER_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

// Membuat fungsi LabelEncoder //
// LabelEncoder berfungsi untuk mengubah data kategorikal (string) menjadi angka //
// Contoh: ["Merah", "Biru", "Merah", "Hijau"] -> [0, 1, 0, 2] //
// Tujuan nya agar data kategorikal bisa diproses oleh algoritma machine learning //
// karena model tidak bisa membaca string secara langsung //

class LabelEncoder {
private:
    std::unordered_map<std::string, int> label2index; // Mapping label -> angka
    std::vector<std::string> index2label;             // Mapping angka -> label (reverse)

public:
    // Constructor default //
    LabelEncoder() {}

    // Fungsi "fit" untuk belajar dari data kategorikal //
    void fit(const std::vector<std::string>& data) {
        label2index.clear();
        index2label.clear();

        int idx = 0;
        for (const auto& label : data) {
            if (label2index.find(label) == label2index.end()) {
                label2index[label] = idx;
                index2label.push_back(label);
                idx++;
            }
        }
    }

    // Fungsi "transform" untuk mengubah string menjadi angka //
    std::vector<int> transform(const std::vector<std::string>& data) const {
        std::vector<int> encoded;
        for (const auto& label : data) {
            auto it = label2index.find(label);
            if (it != label2index.end()) {
                encoded.push_back(it->second);
            } else {
                std::cerr << "Warning: Label \"" << label << "\" tidak dikenali." << std::endl;
                encoded.push_back(-1); // Jika label tidak ada
            }
        }
        return encoded;
    }

    // Fungsi "inverse_transform" untuk mengembalikan angka menjadi string //
    std::vector<std::string> inverse_transform(const std::vector<int>& data) const {
        std::vector<std::string> decoded;
        for (int idx : data) {
            if (idx >= 0 && idx < index2label.size()) {
                decoded.push_back(index2label[idx]);
            } else {
                std::cerr << "Warning: Index " << idx << " tidak valid." << std::endl;
                decoded.push_back("UNKNOWN");
            }
        }
        return decoded;
    }
};

#endif
