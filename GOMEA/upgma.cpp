//
// Created by joe on 16-03-21.
//

#include <cstdlib>
#include <algorithm>
#include "upgma.h"
//#include <limits>
//#include <algorithm>

//For testing purposes
static std::vector<int> lowest_cell(std::vector<std::vector<float>> table){
    float max_cell = std::numeric_limits<float>::max();
    int x, y;

    for(int i=0;i<table.size();i++){
        for(int j=0;j<table[i].size();j++){
            if(table[i][j]<max_cell){
                max_cell=table[i][j];
                x = i;
                y = j;
            }
        }
    }

    return std::vector<int>({x, y});
};

static std::vector<int> highest_cell(std::vector<std::vector<float>> table){
    float min_cell = -std::numeric_limits<float>::max();
    int x, y;

    for(int i=0;i<table.size();i++){
        for(int j=0;j<table[i].size();j++){
            if(table[i][j]>min_cell){
                min_cell=table[i][j];
                x = i;
                y = j;
            }
        }
    }

    return std::vector<int>({x, y});
};

static void join_table(std::vector<std::vector<float>>* table, int a, int b){
    if(b<a){
        a = a + b;
        b = a - b;
        a = a - b;
    }

    std::vector<float> row = {};

    for(int i =0; i<a; i++){
        row.emplace_back(((*table)[a][i] + (*table)[b][i])/2.);
    }
    (*table)[a] = row;

    for(int i =a+1; i<b; i++){
        (*table)[i][a] = ((*table)[i][a]+(*table)[b][i])/2.;
    }

    for(int i =b+1; i<(*table).size(); i++){
        (*table)[i][a] = ((*table)[i][a]+(*table)[i][b])/2.;
        (*table)[i].erase((*table)[i].begin() + b);
    }
    (*table).erase((*table).begin() + b);
};

static void join_labels(std::vector<std::vector<int>>* labels, std::vector<std::vector<int>>* sets, int a, int b){
    if(b<a){
        a = a + b;
        b = a - b;
        a = a - b;
    }

    std::vector<int> new_list = (*labels)[a];
    new_list.insert(new_list.end(), (*labels)[b].begin(), (*labels)[b].end());

    (*labels)[a] = new_list;
    (*sets).emplace_back((*labels)[a]);


    (*labels).erase((*labels).begin() + b);
}


std::vector<std::vector<int>> calc_wpgma(std::vector<std::vector<float>> table, std::vector<std::vector<int>> labels){
    std::vector<std::vector<int>> sets = labels;

    // 2 to remove last set that contains all numbers
    while(labels.size()>2){
        std::vector<int> x_y = highest_cell(table);
        join_table(&table, x_y[0], x_y[1]);
        join_labels(&labels, &sets, x_y[0], x_y[1]);
    }

    return sets;
};

static int DetermineNearestNeighbour(int index, std::vector<std::vector<float>> &S_matrix, std::vector<int> & mpm_number_of_indices, int mpm_length) {
    int i, result;

    result = 0;
    if (result == index) {
        result++;
    }
    for (i = 1; i < mpm_length; i++) {
        if (((S_matrix[index][i] > S_matrix[index][result]) || ((S_matrix[index][i] == S_matrix[index][result]) && (mpm_number_of_indices[i] < mpm_number_of_indices[result]))) && (i != index)) {
            result = i;
        }
    }

    return ( result);
};

std::vector<std::vector<int>> calc_upgma_marco(std::vector<std::vector<float>> &sim_matrix, int number_of_nodes) {
    std::vector<std::vector<int>> FOS;

    std::vector<int> random_order;
    random_order.reserve(number_of_nodes);
    for (int i = 0; i < number_of_nodes; i++)
        random_order.push_back(i);
    random_shuffle(random_order.begin(), random_order.end());

    std::vector<std::vector < int >> mpm(number_of_nodes, std::vector<int>(1));
    std::vector<int> mpm_number_of_indices(number_of_nodes);
    int mpm_length = number_of_nodes;

    for (int i = 0; i < number_of_nodes; i++) {
        mpm[i][0] = random_order[i];
        mpm_number_of_indices[i] = 1;
    }

    /* Initialize LT to the initial MPM */
    FOS.resize(number_of_nodes + number_of_nodes - 1);
    int FOSs_index = 0;
    for (int i = 0; i < mpm_length; i++) {
        FOS[FOSs_index] = std::vector<int>(mpm[i].begin(), mpm[i].end());
        FOSs_index++;
    }

    /* Initialize similarity matrix */
    std::vector<std::vector < float >> S_matrix(number_of_nodes, std::vector<float>(number_of_nodes));
    for (int i = 0; i < mpm_length; i++)
        for (int j = 0; j < mpm_length; j++)
            S_matrix[i][j] = sim_matrix[mpm[i][0]][mpm[j][0]];
    for (int i = 0; i < mpm_length; i++)
        S_matrix[i][i] = 0;

    std::vector<std::vector < int >> mpm_new;
    std::vector<int> NN_chain(number_of_nodes + 2, 0);
    int NN_chain_length = 0;
    short done = 0;

    while (!done) {
        if (NN_chain_length == 0) {
            NN_chain[NN_chain_length] = (int) (std::rand()/RAND_MAX) * mpm_length;
            NN_chain_length++;
        }

        while (NN_chain_length < 3) {
            NN_chain[NN_chain_length] = DetermineNearestNeighbour(NN_chain[NN_chain_length - 1], S_matrix, mpm_number_of_indices, mpm_length);
            NN_chain_length++;
        }

        while (NN_chain[NN_chain_length - 3] != NN_chain[NN_chain_length - 1]) {
            NN_chain[NN_chain_length] = DetermineNearestNeighbour(NN_chain[NN_chain_length - 1], S_matrix, mpm_number_of_indices, mpm_length);
            NN_chain[NN_chain_length] = DetermineNearestNeighbour(NN_chain[NN_chain_length - 1], S_matrix, mpm_number_of_indices, mpm_length);
            if (((S_matrix[NN_chain[NN_chain_length - 1]][NN_chain[NN_chain_length]] == S_matrix[NN_chain[NN_chain_length - 1]][NN_chain[NN_chain_length - 2]])) && (NN_chain[NN_chain_length] != NN_chain[NN_chain_length - 2]))
                NN_chain[NN_chain_length] = NN_chain[NN_chain_length - 2];
            NN_chain_length++;
            if (NN_chain_length > number_of_nodes)
                break;
        }
        int r0 = NN_chain[NN_chain_length - 2];
        int r1 = NN_chain[NN_chain_length - 1];
        int rswap;
        if (r0 > r1) {
            rswap = r0;
            r0 = r1;
            r1 = rswap;
        }
        NN_chain_length -= 3;

        if (r1 < mpm_length) { /* This test is required for exceptional cases in which the nearest-neighbor ordering has changed within the chain while merging within that chain */
            std::vector<int> indices(mpm_number_of_indices[r0] + mpm_number_of_indices[r1]);
            //indices.resize((mpm_number_of_indices[r0] + mpm_number_of_indices[r1]));
            //indices.clear();

            int i = 0;
            for (int j = 0; j < mpm_number_of_indices[r0]; j++) {
                indices[i] = mpm[r0][j];
                i++;
            }
            for (int j = 0; j < mpm_number_of_indices[r1]; j++) {
                indices[i] = mpm[r1][j];
                i++;
            }

            FOS[FOSs_index] = indices;
            FOSs_index++;

            float mul0 = ((float) mpm_number_of_indices[r0]) / ((float) mpm_number_of_indices[r0] + mpm_number_of_indices[r1]);
            float mul1 = ((float) mpm_number_of_indices[r1]) / ((float) mpm_number_of_indices[r0] + mpm_number_of_indices[r1]);
            for (i = 0; i < mpm_length; i++) {
                if ((i != r0) && (i != r1)) {
                    S_matrix[i][r0] = mul0 * S_matrix[i][r0] + mul1 * S_matrix[i][r1];
                    S_matrix[r0][i] = S_matrix[i][r0];
                }
            }

            mpm_new = std::vector<std::vector < int >> (mpm_length - 1);
            std::vector<int> mpm_new_number_of_indices(mpm_length - 1);
            int mpm_new_length = mpm_length - 1;
            for (i = 0; i < mpm_new_length; i++) {
                mpm_new[i] = mpm[i];
                mpm_new_number_of_indices[i] = mpm_number_of_indices[i];
            }

            mpm_new[r0] = std::vector<int>(indices.begin(), indices.end());

            mpm_new_number_of_indices[r0] = mpm_number_of_indices[r0] + mpm_number_of_indices[r1];
            if (r1 < mpm_length - 1) {
                mpm_new[r1] = mpm[mpm_length - 1];
                mpm_new_number_of_indices[r1] = mpm_number_of_indices[mpm_length - 1];

                for (i = 0; i < r1; i++) {
                    S_matrix[i][r1] = S_matrix[i][mpm_length - 1];
                    S_matrix[r1][i] = S_matrix[i][r1];
                }

                for (int j = r1 + 1; j < mpm_new_length; j++) {
                    S_matrix[r1][j] = S_matrix[j][mpm_length - 1];
                    S_matrix[j][r1] = S_matrix[r1][j];
                }
            }

            for (i = 0; i < NN_chain_length; i++) {
                if (NN_chain[i] == mpm_length - 1) {
                    NN_chain[i] = r1;
                    break;
                }
            }

            mpm = mpm_new;
            mpm_number_of_indices = mpm_new_number_of_indices;
            mpm_length = mpm_new_length;

            if (mpm_length == 1)
                done = 1;
        }
    }

    // Remove subsets with all symbols
    FOS.pop_back();

    return FOS;
};