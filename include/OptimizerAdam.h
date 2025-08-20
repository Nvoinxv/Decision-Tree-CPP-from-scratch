#ifndef OPTIMIZER_H 
#define OPTIMIZER_H  

#include <vector> 
#include <cmath>
#include <algorithm>  

class AdamOptimizer { 
private:     
    double alpha;   // learning rate //     
    double beta1;   // momentum //     
    double beta2;   // RMSprop //     
    double epsilon; // epsilon kecil untuk mencegah pembagian 0 //     
    std::vector<double> m; // first moment (mean) //     
    std::vector<double> v; // second moment (variance) //     
    int t; // iterasi //
    
    // Optimisasi //
    double beta1_power; // beta1^t sudah di hitung sebelum nya //
    double beta2_power; // beta2^t sudah di hitung sebelum nya //
    double alpha_scaled; // alpha dengan koreksi bias  //
    bool warmup_phase; // pemanasan learning rate //
    double initial_alpha; // learning rate yang asli //

public:     
    AdamOptimizer(int n, double alpha=0.01, double beta1=0.9, double beta2=0.999, double epsilon=1e-8)         
        : alpha(alpha), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0),
          beta1_power(1.0), beta2_power(1.0), warmup_phase(true), initial_alpha(alpha)
    {         
        m.resize(n, 0.0);         
        v.resize(n, 0.0);
        
        m.reserve(n + 100);
        v.reserve(n + 100);
    }      

    void update(std::vector<double>& theta, const std::vector<double>& g) {         
        t++; // iterasi ke-t //
        
        // untuk menghindari perhitungan berulang //
        beta1_power *= beta1;
        beta2_power *= beta2;
        
        // Pemanasan learning rate sebelum di pakai buat iterasi pertama //
        if (warmup_phase && t <= 10) {
            alpha = initial_alpha * (static_cast<double>(t) / 10.0);
        } else {
            warmup_phase = false;
            alpha = initial_alpha;
        }
        
        // faktor bias koreksi //
        double bias_correction1 = 1.0 - beta1_power;
        double bias_correction2 = 1.0 - beta2_power;
        alpha_scaled = alpha * std::sqrt(bias_correction2) / bias_correction1;
        
        // update vektorisasi dengan gradient //
        for (size_t i = 0; i < theta.size() && i < g.size(); i++) {
            double clipped_g = std::max(-10.0, std::min(10.0, g[i]));
            
            // update m dan v //       
            m[i] = beta1 * m[i] + (1 - beta1) * clipped_g;             
            v[i] = beta2 * v[i] + (1 - beta2) * clipped_g * clipped_g;              
            double m_corrected = m[i];
            double v_corrected = v[i];
            
            // update parameter dengan improve stabilitas angka //
            double denominator = std::sqrt(v_corrected) + epsilon;
            theta[i] -= alpha_scaled * m_corrected / denominator;
        }     
    }
    
    // Penambahan metode utilitas //
    void reset() {
        t = 0;
        beta1_power = 1.0;
        beta2_power = 1.0;
        std::fill(m.begin(), m.end(), 0.0);
        std::fill(v.begin(), v.end(), 0.0);
        warmup_phase = true;
        alpha = initial_alpha;
    }
    
    // Penjadwalan Learning Rate //
    void decay_lr(double factor = 0.9) {
        initial_alpha *= factor;
        alpha = initial_alpha;
    }
    
    // Mendapatkan efisensi learning rate //
    double get_lr() const {
        return alpha_scaled;
    }
    
    // Overloaded update method untuk gradients saja //
    void update(const std::vector<double>& g) {
        t++;
        beta1_power *= beta1;
        beta2_power *= beta2;
    }
};  

#endif
