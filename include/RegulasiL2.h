#ifndef REGULASIL2_H
#define REGULASIL2_H

#include <vector>
#include <cmath>
#include <algorithm>

// Membuat Fungsi menghitung regulasi L2 //
class RegulasiL2 {
private:
    double lambda; // parameter regulasi L2 //
    std::vector<double> w; // bobot model //

public:
    // Konstruktor dengan parameter lambda //
    RegulasiL2(double lambda = 0.1) : lambda(lambda) {}

    // Set bobot model //
    void setWeights(const std::vector<double>& weights) {
        w = weights;
    }

    // Forward pass: hitung total loss dengan regulasi L2 //
    double forward(double original_loss) {
        double l2_penalty = 0.0;
        for (const auto& weight : w) {
            l2_penalty += weight * weight; // w^2 //
        }
        l2_penalty *= (lambda / 2.0); 
        return original_loss + l2_penalty; // L_total = L_original + λ * Σ w^2 //
    }

    // Backward pass: hitung gradien regulasi L2 untuk update bobot //
    std::vector<double> backward(const std::vector<double>& grad_original) {
        std::vector<double> grad_total(grad_original.size(), 0.0);

        for (size_t i = 0; i < grad_original.size() && i < w.size(); i++) {
            // ∂L/∂w = ∂L_original/∂w + λ * w //
            grad_total[i] = grad_original[i] + lambda * w[i];
        }
        return grad_total;
    }

    // Getter untuk lambda //
    double getLambda() const {
        return lambda;
    }

    // Setter untuk lambda //
    void setLambda(double newLambda) {
        lambda = newLambda;
    }
};

#endif
