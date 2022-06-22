//
// Created by joe on 07-10-21.
//

#ifndef DCGP_TREE_ARMA_H
#define DCGP_TREE_ARMA_H

#include <vector>
#include "../Node/Node_arma.h"
#include <armadillo>
#include <iostream>
#include <exception>
#include <stdlib.h>
#include "../Individual/individual_arma.h"

using namespace std;

struct Tree_arma: public Individual_arma {
protected:
    std::vector<int> *use_indices;

    int n_inputs;
    int n_constants;
    bool use_consts;
    int n_outputs;

    int rows;
    int columns;
    int levels_back;

    int max_arity;
    int max_length;

    int batch_size;

    float fitness;
    float penalty;

    bool linear;
    bool normalise;

    bool test_mode;

    std::vector<float> norm;

    std::vector<float> max_erc_value;

    float a;
    float b;

    int node_evaluations = 0;
    int evaluations = 0;

    std::vector<std::string> op_strings;

    std::vector<std::vector<Node_arma>> Node_arma_matrix;
    std::vector<Node_arma> output_Node_armas;

public:
    Tree_arma(int n_inputs, int n_constants, bool use_consts, int n_outputs,
              int rows, int columns, int levels_back, int max_arity,
              int batch_size, bool linear, std::vector<float> max_erc_value, int init_type, bool normalise,
              int max_length, float penalty,
              std::vector<int> *use_indices) {

        Tree_arma::n_inputs = n_inputs;
        Tree_arma::n_constants = n_constants;
        Tree_arma::use_consts = use_consts;
        Tree_arma::n_outputs = n_outputs;

        Tree_arma::rows = rows;
        Tree_arma::columns = columns;
        Tree_arma::levels_back = levels_back;

        Tree_arma::max_arity = max_arity;
        Tree_arma::max_length = max_length;

        Tree_arma::batch_size = batch_size;

        Tree_arma::test_mode = false;

        Tree_arma::fitness = penalty;
        Tree_arma::penalty = penalty;

        Tree_arma::linear = linear;
        Tree_arma::normalise = normalise;
        Tree_arma::max_erc_value = max_erc_value;

        Tree_arma::use_indices = use_indices;
        Tree_arma::op_strings =  {"add", "multiply", "subtract", "analytic_quotient", "pdivide", "max", "min", "pow", "divide", "sin", "cos", "sqrt", "plog", "exp", "square","asin","acos","log"};


        Node_arma_matrix = {};
        output_Node_armas = {};

        a = -1;
        b = -1;

        initialise_individual(init_type);
    }

    virtual int get_batch_size() override{
        return batch_size;
    }

    virtual int get_outputs() override{
        return n_outputs;
    }

    virtual void set_type(int i, int j, bool is_input, bool is_constant, bool is_output){
        Node_arma_matrix[i][j].is_input = is_input;
        Node_arma_matrix[i][j].is_constant = is_constant;
        Node_arma_matrix[i][j].is_output = is_output;
    }



    virtual int get_input_number(int i, int j) override{
        return Node_arma_matrix[i][j].input_number;
    }

    virtual Node_arma get_output_node(int i) override{
        return output_Node_armas[i];
    }

    virtual Node_arma get_node_matrix(int i, int j) override{
        return Node_arma_matrix[i][j];
    }

    virtual float get_fitness() override{
        return fitness;
    }

    virtual int get_n_constants() override{
        return n_constants;
    }

    virtual bool get_test_mode() override{
        return test_mode;
    }

    virtual void set_test_mode(bool test_mode) override{
        Tree_arma::test_mode = test_mode;
    }

    virtual void set_op(int i, int j, int op) override{
        Node_arma_matrix[i][j].op = op;
    }

    virtual void set_input_number(int i, int j, int input_number) override{
        Node_arma_matrix[i][j].input_number = input_number;
    }

    virtual int get_op(int i, int j) override{
        return Node_arma_matrix[i][j].op;
    }

    virtual vector<float> get_norm_var() override{
        return norm;
    }

    virtual void set_linear(bool set_lin) override{
        linear = set_lin;
    }

    virtual void set_arity(int i, int j, int arity) override{
        Node_arma_matrix[i][j].arity = arity;
    }

    virtual void set_Node_arma_matrix(int i, int j, float t, std::vector<int> pos){
        Node_arma_matrix[i][j] = Node_arma(1, pos, max_erc_value, use_indices);
        Node_arma_matrix[i][j].value = t;
    }

    virtual void set_child_positions(int i, int j, std::vector<int> pos){
        Node_arma_matrix[i][j].child_positions.emplace_back(pos);
    }

    virtual void clear_child_positions(int i, int j){
        Node_arma_matrix[i][j].child_positions.clear();
    };

    virtual vector<vector<int>> get_child_positions(int i, int j){
        return Node_arma_matrix[i][j].child_positions;
    }

    virtual int get_node_evaluations(){
        return node_evaluations;
    }

    virtual void clear_node_evaluations(){
        node_evaluations = 0;
    }

    virtual int get_evaluations(){
        return evaluations;
    }

    virtual void clear_evaluations(){
        evaluations = 0;
    }


    virtual bool get_normalise() override{
        return normalise;
    }

    virtual bool get_linear() override{
        return linear;
    }

    virtual vector<vector<int>> get_parents(int i, int j){
        if(j==columns){
            return {};
        }
        else if(!this->Node_arma_matrix[i][j].parent_positions.empty()){
            return this->Node_arma_matrix[i][j].parent_positions;
        }
        else{
            return {};
        }
    }

    // Changes the normalisation factors, tensor is as follows [intercept, slope]
    virtual void change_norm(std::vector<float> norm) override{
        Tree_arma::norm = norm;
    }

    // Changes the linear scaling factors, tensor is as follows [intercept, slope]
    virtual void change_ab(std::vector<float> a_b) override{
        a = a_b[0];
        b = a_b[1];
    }

    virtual Individual_arma* clone() override{
        Individual_arma* new_ind = new Tree_arma(*this);

        return new_ind;
    }

    virtual int get_rows() override{
        return rows;
    }

    virtual int get_columns() override{
        return columns;
    };


    // Returns a string of the active Node_armas. A Node_arma is active when it connected to an output Node_arma
    std::string print_node(Node_arma nod, bool structural_comparison) {
        std::string str = "";

        if (nod.is_input) {
            str += std::string("input_") + std::to_string(nod.input_number);
        } else if (nod.is_constant) {
            if(!structural_comparison) {
                str += std::to_string(nod.value);
            }
            else{
                str += "const";
            }
        } else if (nod.is_output) {
            str += std::string("output_");
            if(linear && normalise){
                str += std::to_string(nod.position[0]) + std::string("=") +
                       std::to_string(norm[0]) + "+" +  std::to_string(norm[1]) +
                       "*" "(" + std::to_string(a) + "+" + std::to_string(b) + "*" + "(" +
                       print_node(Node_arma_matrix[nod.child_positions[0][0]][nod.child_positions[0][1]], structural_comparison) + "))";
            }
            else if(linear){
                str += std::to_string(nod.position[0]) + std::string("=") +
                       std::to_string(a) + "+" + std::to_string(b) + "*" +
                       print_node(Node_arma_matrix[nod.child_positions[0][0]][nod.child_positions[0][1]], structural_comparison);
            }
            else if(normalise){
                str += std::to_string(nod.position[0]) + std::string("=") +
                       std::to_string(norm[0]) + "+" +  std::to_string(norm[1]) +
                       "*" "(" +
                       print_node(Node_arma_matrix[nod.child_positions[0][0]][nod.child_positions[0][1]], structural_comparison) + ")";
            }
            else{
                str += std::to_string(nod.position[0]) + std::string("=") +
                       print_node(Node_arma_matrix[nod.child_positions[0][0]][nod.child_positions[0][1]], structural_comparison);
            }
        } else {

            str += op_strings[nod.op] + std::string("(");
            for (int i = 0; i < nod.arity; i++) {
                str += print_node(get_node_matrix(nod.child_positions[i][0], nod.child_positions[i][1]), structural_comparison);

                if (i == nod.arity - 1) {
                    str += std::string(")");
                } else {
                    str += std::string(",");
                }
            }
        }

        return str;
    }

    arma::mat elementwise_pow(arma::mat base, arma::mat p) {
        arma::mat result;
        result.copy_size(base);
        for (std::size_t i = 0; i < result.n_elem; ++i) {
            result[i] = std::pow(base[i], p[i]);
        }
        return result;
    }

    // TODO: When there is no input in the entire graph the output size is not broadcasted in the correct format?
    // Returns the output tensor of a Node_arma given an input tensor x and a batch size
    arma::mat node_forward(Node_arma nod, arma::mat x, int batch_size){
        vector<int> pos = nod.position;
        if (nod.is_input) {
            node_evaluations++;
            return x.col(nod.input_number);
        }
        else if (nod.is_constant) {
            node_evaluations++;
            return arma::ones(batch_size,1)* nod.value;
        }
        else if (nod.is_output) {

            arma::mat forward = node_forward(Node_arma_matrix[nod.child_positions[0][0]][nod.child_positions[0][1]], x, batch_size);

            node_evaluations++;
            return forward;
        }
        else {
            std::vector<arma::mat> c;
            for (int i = 0; i < nod.arity; i++) {
                c.emplace_back(node_forward(Node_arma_matrix[nod.child_positions[i][0]][nod.child_positions[i][1]],x, batch_size));
            }

            switch(nod.op){
                case 0:
                    x = c[0] + c[1];
                    break;
                case 1:
                    x = c[0] % c[1];
                    break;
                case 2:
                    x = c[0] - c[1];
                    break;
                case 3:
                    x = c[0] / arma::sqrt(1. + arma::square(c[1]));
                    break;
                case 4:
                    x = arma::sign(c[1])%(c[0] / (arma::abs(c[1]) + 10e-6));
                    break;
                case 5:
                    x = arma::max(c[0], c[1]);
                    break;
                case 6:
                    x = arma::min(c[0], c[1]);
                    break;
                case 7:
                    x = elementwise_pow(c[0],c[1]);
                    break;
                case 8:
                    x = c[0] / c[1];
                    break;
                case 9:
                    x = arma::sin(c[0]);
                    break;
                case 10:
                    x = arma::cos(c[0]);
                    break;
                case 11:
                    x = arma::sqrt(c[0]);
                    break;
                case 12:
                    x = arma::log(arma::abs(c[0]) + 10e-6);
                    break;
                case 13:
                    x = arma::exp(c[0]);
                    break;
                case 14:
                    x = arma::square(c[0]);
                    break;
                case 15:
                    x = arma::asin(c[0]);
                    break;
                case 16:
                    x = arma::acos(c[0]);
                    break;
                case 17:
                    x = arma::log(c[0]);
                    break;
            }

            node_evaluations++;
            return x;
        }
    }

    // Returns whether a Node_arma is used in the graph by traversing the graph from each output Node_arma
    virtual std::vector<Node_arma> active_node(Node_arma nod) override{
        std::vector<Node_arma> o = {};
        if (nod.is_input) {
            o.emplace_back(nod);
            return o;
        } else if (nod.is_constant) {
            o.emplace_back(nod);
            return o;
        } else if (nod.is_output) {
            o.emplace_back(nod);

            std::vector<Node_arma> o2 = active_node(Node_arma_matrix[nod.child_positions[0][0]][nod.child_positions[0][1]]);
            o.insert(o.end(), o2.begin(), o2.end());

            return o;
        } else {
            o.emplace_back(nod);
            if(nod.child_positions.size()>0) {
                for (int i = 0; i < nod.arity; i++) {
                    std::vector<Node_arma> o2 = active_node(
                            Node_arma_matrix[nod.child_positions[i][0]][nod.child_positions[i][1]]);
                    o.insert(o.end(), o2.begin(), o2.end());
                }
            }

            return o;
        }
    }

    // Returns active Node_arma in graph for each output
    virtual std::vector<Node_arma> get_active_nodes() override{
        std::vector<Node_arma> active_Node_armas = {};
        for (int i = 0; i < n_outputs; i++) {
            std::vector<Node_arma> o = active_node(output_Node_armas[i]);
            active_Node_armas.insert(active_Node_armas.end(), o.begin(), o.end());
            o.clear();
        }
        return active_Node_armas;
    }

    // Forward pass through graph for each output
    virtual arma::mat forward(arma::mat x, int batch_size) override{
        std::vector<arma::mat> v;
        for (int i = 0; i < n_outputs; i++) {
            v.emplace_back(node_forward(output_Node_armas[i], x, batch_size));
        }
        //TODO
        return v[0];
    }

    // Prints Node_armas using the child positions
    virtual std::string print_active_nodes(bool structural_comparison) override{
        std::string str = "";
        for (int i = 0; i < n_outputs; i++) {
            str += print_node(output_Node_armas[i], structural_comparison);
        }
        return str;
    }

//    vector<Node_arma> get_child_nodes(Node_arma node){
//        vector<Node_arma> nodes = {};
//        if(!node.is_input && !node.is_constant){
//            vector<vector<int>> child_positions = {};
//            for(auto child_position:node.child_positions){
//                nodes.emplace_back();
//            }
//        }
//    }

    virtual int get_semantic_duplicates(arma::mat train_x){
        std::vector<arma::mat> outputs = {};
        std::vector<Node_arma> output_nodes = {};
        std::vector<Node_arma> active_nodes = get_active_nodes();
        arma::mat x(1000, n_inputs, arma::fill::randu);
        x = x*(max_erc_value[1]-max_erc_value[0]) + max_erc_value[0];

        arma::mat x2 = arma::join_cols(x, train_x);

        for(auto node:active_nodes){
            if(!node.is_output && !node.is_constant && !node.is_input){
                outputs.emplace_back(node_forward(node, x2, x2.n_rows));
                output_nodes.emplace_back(node);
            }
        }



        int count = 0;

        for(int i =0; i<outputs.size(); i++){
            for(int j=i+1; j<outputs.size();j++){
                if(approx_equal(outputs[i], outputs[j], "absdiff", 0.0000000001)){
//
                    cout<<print_node(output_nodes[i], false)<<endl;
                    cout<<print_node(output_nodes[j], false)<<endl;
//                    cout<<output_nodes[i].op<<" "<<output_nodes[i].position[0]<<","<<output_nodes[i].position[1]<<" "<<output_nodes[j].op<<" "<<output_nodes[j].position[0]<<","<<output_nodes[j].position[1]<<endl;
                    count++;
                }
            }
        }

        return count;
    }

    // Evaluate the fitness of individual. Int that is returned can optionally be used for debuggging
    virtual int evaluate(arma::mat X, arma::mat y) override{
        evaluations += 1;
        fitness = 0.0;

        if(max_length > 0  && get_active_nodes().size()>max_length){
            fitness += penalty;
            return 0;
        }

        arma::mat output;
        try {
            output = forward(X, X.n_rows);
        }
        catch (std::exception &e) {
            fitness += penalty;
            return 1;
        };

        if(normalise){
            output = norm[0] + norm[1] * output;
        }

        if (linear) {
            // In test mode predefined a and b are used
            if(!test_mode){
                std::vector<float> a_b = get_ab(X, y, output);
                output = a_b[0] + a_b[1] * output;
            }
            else{
                output = a + b * output;
            }


        }

        if ((output).has_nan() || output.has_inf()) {
            fitness += penalty;
            return 5;
        }

        arma::mat loss = arma::mean(arma::square(output-y));

        fitness += loss.at(0);


        return 4;
    }

    void grow(){
        int height = columns;
        Node_arma_matrix.clear();

        for(int i=0; i<pow(2,height-1);i++){
            Node_arma_matrix.emplace_back(std::vector<Node_arma>({}));
            for (int j = 0; j < height - ceil(log2(i+1)); j++) {
                std::vector<int> pos = {i, j};
                //Sample from terminal set
                if(j==0) {
                    int r;
                    if(Tree_arma::use_consts){
                        r = std::rand() % (n_inputs + 1);
                    }
                    else{
                        r = std::rand() % n_inputs;
                    }

                    if(r<n_inputs){
                        Node_arma_matrix[i].emplace_back(0, pos, max_erc_value, use_indices);
                        Node_arma_matrix[i][j].set_input_number(int(r));
                    }
                    else{
                        Node_arma_matrix[i].emplace_back(1, pos, max_erc_value, use_indices);
                    }
                }
                else {
                    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                    if(r>0.5) {
                        Node_arma_matrix[i].emplace_back(3, pos, max_erc_value, use_indices);
                    }
                    else {
                        if(Tree_arma::use_consts){
                            r = std::rand() % (n_inputs + 1);
                        }
                        else{
                            r = std::rand() % n_inputs;
                        }
                        if(r<n_inputs){
                            Node_arma_matrix[i].emplace_back(0, pos, max_erc_value, use_indices);
                            Node_arma_matrix[i][j].set_input_number(int(r));
                        }
                        else{
                            Node_arma_matrix[i].emplace_back(1, pos, max_erc_value, use_indices);
                        }
                    }
                }
            }
        }
    }

    void full(){
        int height = columns;
        Node_arma_matrix.clear();

        for(int i=0; i<pow(2,height-1);i++){
            Node_arma_matrix.emplace_back(std::vector<Node_arma>({}));
            for (int j = 0; j < height - ceil(log2(i+1)); j++) {
                std::vector<int> pos = {i, j};
                if(j==0) {
                    int r;
                    if(Tree_arma::use_consts){
                        r = std::rand() % (n_inputs + 1);
                    }
                    else{
                        r = std::rand() % n_inputs;
                    }

                    if(r<n_inputs){
                        Node_arma_matrix[i].emplace_back(0, pos, max_erc_value, use_indices);
                        Node_arma_matrix[i][j].set_input_number(int(r));
                    }
                    else{
                        Node_arma_matrix[i].emplace_back(1, pos, max_erc_value, use_indices);
                    }
                }
                else {
                    Node_arma_matrix[i].emplace_back(3, pos, max_erc_value, use_indices);
                }
            }
        }
    }

    virtual void initialise_individual(int type) override {
        int height = columns;

        if(type==0){
            grow();
        }
        else{
            full();
        }

        for(int i=0; i<pow(2,height-1);i++){
            for (int j = 1; j < height - ceil(log2(i+1)); j++) {
                for (int k = 0; k < Tree_arma::max_arity; k++) {

                    //TODO: wrong? Why not just emplace back directly?
                    Node_arma_matrix[i][j].child_positions.emplace_back(Node_arma_matrix[i][j].position);
                    Node_arma_matrix[i][j].child_positions[k] = Node_arma_matrix[std::min(2*i+k, int(pow(2,height-1))-1)][j-1].position;

                    Node_arma_matrix[std::min(2*i+k, int(pow(2,height-1))-1)][j-1].parent_positions.emplace_back(Node_arma_matrix[i][j].position);
                    //                    Node_arma_matrix[i][j].child_positions.emplace_back(Node_arma_matrix[std::min(2*i+k, int(pow(2,height-1))-1)][j-1].position);
                }

            }
        }

        output_Node_armas.clear();
        //TODO: may be incorrect
        std::vector<int> pos = {0, height};
        output_Node_armas.emplace_back(2, pos, max_erc_value, use_indices);

        output_Node_armas[0].child_positions.emplace_back(Node_arma_matrix[0][height-1].position);
        Node_arma_matrix[0][height-1].parent_positions.emplace_back(output_Node_armas[0].position);

    }
};

#endif //DCGP_TREE_ARMA_H