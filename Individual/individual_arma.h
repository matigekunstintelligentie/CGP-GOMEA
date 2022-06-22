//
// Created by joe on 15-03-21.
//

#ifndef DCGP_INDIVIDUAL_ARMA_H
#define DCGP_INDIVIDUAL_ARMA_H

#include <vector>
#include "../Node/Node_arma.h"
#include <iostream>
#include <exception>
#include <stdlib.h>
#include <armadillo>

struct Individual_arma {
public:
    Individual_arma(){};

    virtual ~Individual_arma() = default;

    virtual void initialise_individual(int type) = 0;

    virtual int evaluate(arma::mat X, arma::mat y) = 0;

    virtual Individual_arma* clone() = 0;

    virtual std::string print_active_nodes(bool) = 0;

    virtual std::vector<Node_arma> get_active_nodes() = 0;

    virtual arma::mat forward(arma::mat x, int batch_size) = 0;

    //GETTERS
    virtual int get_batch_size() = 0;
    virtual int get_outputs() = 0;
    virtual int get_rows() = 0;
    virtual int get_columns() = 0;
    virtual int get_n_constants() = 0;
    virtual bool get_test_mode() = 0;
    virtual void set_test_mode(bool test_mode) = 0;
    virtual float get_fitness() = 0;
    virtual Node_arma get_input_node(int i){
        throw std::runtime_error("Abstract get_input_node function called");
    };
    virtual bool get_normalise() = 0;
    virtual bool get_linear() = 0;

    virtual void change_norm(std::vector<float> norm) = 0;

    virtual void change_ab(std::vector<float> ab) = 0;

    virtual Node_arma get_output_node(int i) = 0;

    virtual Node_arma get_node_matrix(int i, int j) = 0;

    virtual int get_op(int i, int j) = 0;

    virtual std::vector<float> get_norm_var() = 0;

    virtual std::vector<int> get_node_matrix_child_pos(int,int,int){
        throw std::runtime_error("Abstract get_node_matrix_child_pos function called");
    };

    virtual std::vector<int> get_constant_pos(int){
        throw std::runtime_error("Abstract get_constant_pos function called");
    };

    virtual float get_ERC(std::vector<int> pos){
        throw std::runtime_error("Abstract get_ERC function called");
    };

    virtual void set_ERC(std::vector<int> pos, float ERC){
        throw std::runtime_error("Abstract get_ERC function called");
    };

    virtual void set_node_matrix_child_pos(int,int,int,std::vector<int>){
        throw std::runtime_error("Abstract set_node_matrix_child_pos function called");
    };

    virtual std::vector<int> get_output_child_pos(int){
        throw std::runtime_error("Abstract get_output_child_pos function called");
    };

    virtual void set_output_child_pos(int, std::vector<int>){
        throw std::runtime_error("Abstract set_output_child_pos function called");
    };

    virtual void set_linear(bool setlin) = 0;

    virtual void set_input_number(int i, int j, int input_number) = 0;

    virtual void set_op(int i, int j, int op) = 0;

    virtual void set_arity(int i, int j, int arity) = 0;

    virtual std::vector<Node_arma> active_node(Node_arma nod) = 0;

    virtual bool mutate(std::string active_nodes){
        throw std::runtime_error("Abstract mutate function called");
    };

    virtual void set_node(int i, float t, std::vector<int> pos){};

    virtual void set_node_matrix(int i, int j, float t, std::vector<int> pos){};

    virtual void set_child_positions(int i, int j, std::vector<int> pos){};

    virtual std::vector<std::vector<int>> get_child_positions(int i, int j){
        throw std::runtime_error("Abstract get_child_positions function called");
    };

    virtual void set_type(int i, int j, bool is_input, bool is_constant, bool is_output){};

    virtual void clear_child_positions(int i, int j){};

    virtual std::vector<std::vector<int>> get_parents(int i, int j){
        throw std::runtime_error("Abstract get_parents function called");
    };

    virtual int get_input_number(int,int){
        throw std::runtime_error("Abstract get_input_number function called");
    };

    virtual int get_node_evaluations(){
        throw std::runtime_error("Abstract get_node_evaluations function called");
    };

    virtual void clear_node_evaluations(){
        throw std::runtime_error("Abstract clear_node_evaluations function called");
    };

    virtual int get_evaluations(){
        throw std::runtime_error("Abstract get_node_evaluations function called");
    };

    virtual void clear_evaluations(){
        throw std::runtime_error("Abstract clear_node_evaluations function called");
    };

    void mutate_until_active(){
        bool active_node_mutated = false;
        // get active nodes only once
        std::string active_nodes = print_active_nodes(false);
        while (active_node_mutated != true) {
            active_node_mutated = mutate(active_nodes);
        }
    }

    // Returns the number of unique nodes per output. Used as statistic
    std::vector<std::vector<int>> get_node_number(bool unique){
        std::vector<std::vector<int>> node_numbers;

        std::vector<std::vector<int>> unique_pos;

        for (int i = 0; i < get_outputs(); i++) {
            std::vector<Node_arma> o = active_node(get_output_node(i));
            std::vector<int> inp = {0,0,0};

            for(int j=0; j<o.size();j++){
                if(unique){
                    if(std::find(unique_pos.begin(), unique_pos.end(), o[j].position) != unique_pos.end()){
                        continue;
                    }
                    unique_pos.emplace_back(o[j].position);
                }

                if(o[j].is_input){
                    inp[0] += 1;
                }
                if(o[j].is_constant){
                    inp[1] += 1;
                }
                if(o[j].is_output){
                    continue;
                }
                if(!o[j].is_input && !o[j].is_constant && !o[j].is_output){
                    inp[2] += 1;
                }
            }
            node_numbers.emplace_back(inp);
        }

        return node_numbers;
    };

    std::vector<float> ab_calc(arma::mat tensorX, arma::mat tensory, arma::mat p){
        std::vector<float> res;

        arma::mat mean_p = arma::mean(p);
        arma::mat var_terms_p = p - mean_p.at(0);
        arma::mat denom_p = arma::sum(arma::square(var_terms_p));

        arma::mat y = tensory;
        arma::mat mean_y = arma::mean(y);
        arma::mat var_terms_y = y - mean_y.at(0);

        if (denom_p.at(0) != 0) {
            arma::mat b = arma::sum(var_terms_y % var_terms_p) / denom_p;
            arma::mat a = mean_y - b * mean_p.at(0);
            res.emplace_back(a.at(0));
            res.emplace_back(b.at(0));
        } else {
            arma::mat b = arma::mat({0.0});
            arma::mat a = mean_p;
            res.emplace_back(a.at(0));
            res.emplace_back(b.at(0));
        }

        change_ab(res);

        return res;
    }

    // Returns the a and b factors for linear scaling
    std::vector<float> get_ab(arma::mat tensorX, arma::mat tensory, arma::mat output){
        return ab_calc(tensorX, tensory, output);
    }

    // Returns the a and b factors for linear scaling
    std::vector<float> get_ab(arma::mat tensorX, arma::mat tensory){
        arma::mat output = this->forward(tensorX, tensorX.n_rows);
        std::vector<float> norm_var = get_norm_var();

        if(get_normalise()){
            output = norm_var[0] + norm_var[1] * output;
        }

        return ab_calc(tensorX, tensory, output);
    }

    virtual int get_semantic_duplicates(arma::mat train_x){
        throw std::runtime_error("Abstract get_semantic_duplicates function called");
    };

};


#endif //DCGP_INDIVIDUAL_ARMA_H