//
// Created by joe on 06-10-21.
//

#ifndef DCGP_NODE_ARMA_H
#define DCGP_NODE_ARMA_H

#include <armadillo>
#include <vector>
#include <iostream>


struct Node_arma{
public:
    Node_arma(int node_type, std::vector<int> position, std::vector<float> max_erc_value, std::vector<int> *use_indices) {
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        value = max_erc_value[0] + (max_erc_value[1]-max_erc_value[0])*r;
        initialise_operator(use_indices);

        switch (node_type) {
            case 0:
                is_input = true;
                break;
            case 1:
                is_constant = true;
                break;
            case 2:
                is_output = true;
                break;
            case 3:
                break;
            default:
                throw std::invalid_argument("Received wrong node type");
        }

        for (int i = 0; i < 2; ++i) {
            Node_arma::position[i] = position[i];
        }
    }

    arma::mat cached_value;

    bool is_input = false;
    bool is_constant = false;
    bool is_output = false;

    int input_number = -1;

    std::vector<int> position = {0, 0};
    float value;
    int arity;

    std::vector<std::vector<int>> child_positions;
    std::vector<std::vector<int>> parent_positions;
    int op = -1;

    bool operator==(const Node_arma &rhs) const {
        return rhs.position == position;
    }

    void set_input_number(int inp){
        input_number = inp;
    }

    void initialise_operator(std::vector<int> *use_indices) {
        int r = rand() % use_indices->size();
        op = use_indices->at(r);
        if(op>8){
            arity = 1;
        }
        else{
            arity = 2;
        }
    }

    // Mutates the operator based on which indices can be used.
    // All operator above index 7 have an arity of 1
    void mutate_operator(std::vector<int> *use_indices) {
        // Choose different op number when there is a collision
        int r = rand() % use_indices->size();
        while (r == op) {
            r = rand() % use_indices->size();
        }
        op = use_indices->at(r);

        //TODO:hard-coded atm:(
        if(op>8){
            arity = 1;
        }
        else{
            arity = 2;
        }
    }
};

#endif //DCGP_NODE_ARMA_H
