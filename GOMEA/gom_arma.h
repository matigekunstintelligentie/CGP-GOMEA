

//
// Created by joe on 07-10-21.
//

#include "fos_arma.h"
#include "../Individual/individual_arma.h"

#ifndef DCGP_GOM_ARMA_H
#define DCGP_GOM_ARMA_H


class gom_arma {
public:
    gom_arma(bool fix_output_nodes, int max_constants, bool tree_mode);



    Individual_arma* fast_mix(int ind, std::vector<std::vector<int>> fos, int seed, arma::mat *X, arma::mat *y, std::vector<Individual_arma* > pop, int *rows, int *columns, int *outputs, int *fos_truncation);

    void fast_override_nodes(Individual_arma* individual, Individual_arma* individual1, std::vector<int> *subset, int rows, int columns, int n_outputs);

    void fast_override_nodes_gp(Individual_arma* individual, Individual_arma* individual1, std::vector<int> *subset, int columns);


    bool fix_output_nodes;
    int max_constants;
    bool tree_mode;
};


#endif //DCGP_GOM_ARMA_H