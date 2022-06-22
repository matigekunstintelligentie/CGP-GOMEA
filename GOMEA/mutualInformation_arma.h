//
// Created by joe on 07-10-21.
//

#ifndef DCGP_MUTUALINFORMATION_ARMA_H
#define DCGP_MUTUALINFORMATION_ARMA_H

#include <vector>
#include "../Individual/individual_arma.h"
#include <unordered_map>
#include <boost/functional/hash.hpp>

class MutualInformation_arma {
public:
    bool tree;
    std::map<int, int> convert_op;

    MutualInformation_arma(bool tree, std::vector<int> ops);

    int fast_number_to_node_int (int gene, int rows, Individual_arma* ind);

    std::vector<int> fast_number_to_node_int_gp (int gene, int rows, Individual_arma *ind, int max_ops, int max_inputs);

    std::vector<int> unified_number_to_node_int(int gene, int rows, int columns, Individual_arma *ind, int max_ops, int max_inputs);

    void fast_calc_mutual_information(int subset_size, std::vector<Individual_arma*>, std::vector<std::vector<float>> *mi, bool MI_distribution_adjustments, std::vector<std::vector<float>> *MI_adjustments, int max_ops, int max_inputs, int max_constants,bool normalise);
};


#endif //DCGP_MUTUALINFORMATION_ARMA_H
