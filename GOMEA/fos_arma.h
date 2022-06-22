//
// Created by joe on 07-10-21.
//

#ifndef DCGP_FOS_ARMA_H
#define DCGP_FOS_ARMA_H

#include <vector>
#include "../Individual/individual_arma.h"

class Fos_arma {
public:
    Fos_arma(bool apply_MI_adjustments, int max_constants, bool tree_mode);
    void build_fos(int fos_type, int rows, int columns, int n_outputs, std::vector<Individual_arma*>, std::vector<std::vector<float>> *MI_adjustments, int max_ops, int max_inputs, bool normalise_MI, std::vector<int> ops, bool mix_ERCs);
    void build_univariate_fos(int univariate_size);
    void build_random_fos(int univariate_size);
    void build_random_marco(int uni_size);
    void replace_MI(std::vector<std::vector<float>> M, int size);
    void build_mutual_information_fos(std::vector<std::vector<float>>*, int univariate_size);

    std::vector<std::vector<int>> subsets;
    bool apply_MI_adjustments;
    int max_constants;
    bool tree_mode;
};


#endif //DCGP_FOS_ARMA_H
