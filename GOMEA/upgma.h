
//
// Created by joe on 16-03-21.
//

#ifndef DCGP_UPGMA_H
#define DCGP_UPGMA_H


#include <vector>


static void join_labels(std::vector<std::vector<int>>* labels, std::vector<std::vector<int>>* sets, int a, int b);

static void join_table(std::vector<std::vector<float>>*, int, int);

std::vector<std::vector<int>> calc_wpgma(std::vector<std::vector<float>>, std::vector<std::vector<int>>);

static std::vector<int> highest_cell(std::vector<std::vector<float>>);

static std::vector<int> lowest_cell(std::vector<std::vector<float>>);

std::vector<std::vector<int>> calc_upgma_marco(std::vector<std::vector<float>> &M, int);
static int DetermineNearestNeighbour(int index, std::vector<std::vector<float>> &S_matrix, std::vector<int> & mpm_number_of_indices, int mpm_length);

#endif //DCGP_UPGMA_H