//
// Created by joe on 7/6/21.
//

#ifndef DCGP_CGP_ARMA_H
#define DCGP_CGP_ARMA_H

#include <vector>
#include "../Node/Node_arma.h"
#include <iostream>
#include <exception>
#include <stdlib.h>
#include "../Individual/individual_arma.h"

using namespace std;

struct CGP_arma: public Individual_arma {
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

    int node_evaluations = 0;
    int evaluations = 0;

    std::vector<float> norm;

    std::vector<float> max_erc_value;

    float a;
    float b;

    float gradient_clip;
    float inject_noise;

    std::vector<std::string> op_strings;

    std::vector<std::vector<Node_arma>> node_matrix;
    std::vector<Node_arma> input_nodes;
    std::vector<Node_arma> output_nodes;
public:
    CGP_arma(int n_inputs, int n_constants, bool use_consts, int n_outputs,
             int rows, int columns, int levels_back, int max_arity,
             int batch_size, bool linear, std::vector<float> max_erc_value, int init_type, bool normalise,
             int max_length, float penalty, std::vector<int> *use_indices){

        CGP_arma::n_inputs = n_inputs;
        CGP_arma::n_constants = n_constants;
        CGP_arma::use_consts = use_consts;
        CGP_arma::n_outputs = n_outputs;

        CGP_arma::rows = rows;
        CGP_arma::columns = columns;
        CGP_arma::levels_back = levels_back;

        CGP_arma::max_arity = max_arity;
        CGP_arma::max_length = max_length;

        CGP_arma::batch_size = batch_size;

        CGP_arma::test_mode = false;

        CGP_arma::fitness = penalty;
        CGP_arma::penalty = penalty;

        CGP_arma::linear = linear;
        CGP_arma::normalise = normalise;
        CGP_arma::max_erc_value = max_erc_value;

        CGP_arma::gradient_clip = gradient_clip;
        CGP_arma::inject_noise = inject_noise;

        CGP_arma::use_indices = use_indices;
        CGP_arma::op_strings = {"add", "multiply", "subtract", "analytic_quotient", "pdivide", "max", "min", "pow", "divide", "sin", "cos", "sqrt", "plog", "exp", "square","asin","acos","log"};


        node_matrix = {};
        input_nodes = {};
        output_nodes = {};

        a = -1;
        b = -1;

        initialise_individual(init_type);


    }

    virtual std::vector<int> get_constant_pos(int row) override{
        return input_nodes[n_inputs + row - 1].position;
    }

    virtual float get_ERC(std::vector<int> pos) override{
        return input_nodes[pos[0]].value;
    }

    virtual void set_ERC(std::vector<int> pos, float ERC){
        input_nodes[pos[0]].value = ERC;
    }

    virtual int get_batch_size() override{
        return batch_size;
    }

    virtual int get_outputs() override{
        return n_outputs;
    }

    virtual Node_arma get_input_node(int i) override{
        return input_nodes[i];
    }

    virtual Node_arma get_output_node(int i) override{
        return output_nodes[i];
    }

    virtual Node_arma get_node_matrix(int i, int j) override{
        return node_matrix[i][j];
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

    virtual void set_input_number(int i, int j, int input_number) override{

    }

    virtual void set_test_mode(bool test_mode) override{
        CGP_arma::test_mode = test_mode;
    }

    virtual int get_node_evaluations(){
        return node_evaluations;
    }

    virtual int get_evaluations(){
        return evaluations;
    }

    virtual void set_op(int i, int j, int op) override{
        node_matrix[i][j].op = op;
    }

    virtual int get_op(int i, int j) override{
        return node_matrix[i][j].op;
    }

    virtual vector<float> get_norm_var() override{
        return norm;
    }

    virtual void set_linear(bool set_lin) override{
        linear = set_lin;
    }

    virtual void set_arity(int i, int j, int arity) override{
        node_matrix[i][j].arity = arity;
    }

    virtual void set_node(int i, float t, std::vector<int> pos){
        input_nodes[i] = Node_arma(1, pos, max_erc_value, use_indices);
        input_nodes[i].value = t;
    }

    //Copy constructor. Important note: without copy constructor after mutation the children will point to the wrong nodes!
    virtual Individual_arma* clone() override{

        Individual_arma* new_ind = new CGP_arma(*this);

        return new_ind;
    }

    virtual bool get_normalise() override{
        return normalise;
    }

    virtual bool get_linear() override{
        return linear;
    }


    virtual void set_node_matrix_child_pos(int r, int c, int child, vector<int> pos) override{
        node_matrix[r][c].child_positions[child] = pos;
    }

    virtual vector<int> get_node_matrix_child_pos(int r, int c, int child) override{
        return node_matrix[r][c].child_positions[child];
    }

    virtual void set_output_child_pos(int output_n, vector<int> pos) override{
        output_nodes[output_n].child_positions[0] = pos;
    }

    virtual vector<int> get_output_child_pos(int output_n) override{
        return output_nodes[output_n].child_positions[0];
    }

    // Changes the normalisation factors, tensor is as follows [intercept, slope]
    virtual void change_norm(std::vector<float> norm) override{
        CGP_arma::norm = norm;
    }

    // Changes the linear scaling factors, tensor is as follows [intercept, slope]
    virtual void change_ab(std::vector<float> a_b) override{
        a = a_b[0];
        b = a_b[1];
    }

    virtual int get_rows() override{
        return rows;
    }

    virtual int get_columns() override{
        return columns;
    };



    virtual void clear_node_evaluations(){
        node_evaluations = 0;
    }

    virtual void clear_evaluations(){
        evaluations = 0;
    }

    virtual vector<vector<int>> get_parents(int i, int j){
        if(j==(columns+1)){
            return {};
        }
        else if(j==0 && !this->input_nodes[i].parent_positions.empty()){
            return this->input_nodes[i].parent_positions;
        }
        else if(!this->node_matrix[i][j-1].parent_positions.empty()){
            return this->node_matrix[i][j-1].parent_positions;
        }
        else{
            return {};
        }
    }

    // Returns a string of the active nodes. A node is active when it connected to an output node
    std::string print_node(Node_arma nod, bool structural_comparison) {
        std::string str = "";

        if (nod.is_input) {
            str += std::string("input_") + std::to_string(nod.position[0]);
        } else if (nod.is_constant) {
            if(!structural_comparison) {
                str += std::to_string(nod.value);
            }
            else{
                str += "const";
            }
        } else if (nod.is_output) {
            if (nod.child_positions[0][1] == 0) {
                if(linear && normalise){
                    str += std::string("output_") + std::to_string(nod.position[0]) + std::string("=") +
                           std::to_string(norm[0]) + "+" +  std::to_string(norm[1]) +
                           "*" "(" + std::to_string(a) + "+" + std::to_string(b) + "*" + "(" +
                           print_node(input_nodes[nod.child_positions[0][0]], structural_comparison) + "))";
                }
                else if(linear){
                    str += std::string("output_") + std::to_string(nod.position[0]) + std::string("=") +
                           std::to_string(a) + "+" + std::to_string(b) + "*" + "(" +
                           print_node(input_nodes[nod.child_positions[0][0]], structural_comparison) + ")";
                }
                else if(normalise){
                    str += std::string("output_") + std::to_string(nod.position[0]) + std::string("=") +
                           std::to_string(norm[0]) + "+" +  std::to_string(norm[1]) +
                           "*" "(" +
                           print_node(input_nodes[nod.child_positions[0][0]], structural_comparison) + ")";
                }
                else{
                    str += std::string("output_") + std::to_string(nod.position[0]) + std::string("=") +
                           print_node(input_nodes[nod.child_positions[0][0]], structural_comparison);
                }

            } else {
                if(linear && normalise){
                    str += std::string("output_") + std::to_string(nod.position[0]) + std::string("=") +
                           std::to_string(norm[0]) + "+" +  std::to_string(norm[1]) +
                           "*" "(" + std::to_string(a) + "+" + std::to_string(b) + "*" + "(" +
                           print_node(node_matrix[nod.child_positions[0][0]][nod.child_positions[0][1] - 1], structural_comparison) + "))";
                }
                else if(linear){
                    str += std::string("output_") + std::to_string(nod.position[0]) + std::string("=") +
                           std::to_string(a) + "+" + std::to_string(b) + "*" +
                           print_node(node_matrix[nod.child_positions[0][0]][nod.child_positions[0][1] - 1], structural_comparison);
                }
                else if(normalise){
                    str += std::string("output_") + std::to_string(nod.position[0]) + std::string("=") +
                           std::to_string(norm[0]) + "+" +  std::to_string(norm[1]) +
                           "*" "(" +
                           print_node(node_matrix[nod.child_positions[0][0]][nod.child_positions[0][1] - 1], structural_comparison) + ")";
                }
                else{
                    str += std::string("output_") + std::to_string(nod.position[0]) + std::string("=") +
                           print_node(node_matrix[nod.child_positions[0][0]][nod.child_positions[0][1] - 1], structural_comparison);
                }

            }
        } else {
            str += op_strings[nod.op] + std::string("(");
            for (int i = 0; i < nod.arity; i++) {
                if (nod.child_positions[i][1] == 0) {
                    str += print_node(input_nodes[nod.child_positions[i][0]], structural_comparison);
                } else {
                    str += print_node(node_matrix[nod.child_positions[i][0]][nod.child_positions[i][1] - 1], structural_comparison);
                }
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

    // Returns the output tensor of a node given an input tensor x and a batch size
    arma::mat node_forward(Node_arma nod, arma::mat x, int batch_size){
        if (nod.is_input) {
            node_evaluations++;
            return x.col(nod.position[0]);
        }
        else if (nod.is_constant) {
            node_evaluations++;
            return arma::ones(batch_size,1)* nod.value;

        }
        else if (nod.is_output) {
            if (nod.child_positions[0][1] == 0) {
                arma::mat forward = node_forward(input_nodes[nod.child_positions[0][0]], x, batch_size);
                node_evaluations++;
                return forward;
            }
            else{
                arma::mat forward = node_forward(node_matrix[nod.child_positions[0][0]][nod.child_positions[0][1] - 1], x, batch_size);
                node_evaluations++;
                return forward;
            }
        }
        else {
            std::vector<arma::mat> c;
            for (int i = 0; i < nod.arity; i++) {
                if (nod.child_positions[i][1] == 0) {
                    c.emplace_back(node_forward(input_nodes[nod.child_positions[i][0]],x, batch_size));
                }
                else{
                    c.emplace_back(node_forward(node_matrix[nod.child_positions[i][0]][nod.child_positions[i][1]-1],x, batch_size));
                }
            }
            // TODO: make function in inidividual_arma class
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

    // TODO: Is the active graph correct when the arity is 1?
    // Returns whether a node is used in the graph by traversing the graph from each output node
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
            if(nod.child_positions[0][1]==0){
                std::vector<Node_arma> o2 = active_node(input_nodes[nod.child_positions[0][0]]);
                o.insert(o.end(), o2.begin(), o2.end());
            }
            else{
                std::vector<Node_arma> o2 = active_node(node_matrix[nod.child_positions[0][0]][nod.child_positions[0][1] - 1]);
                o.insert(o.end(), o2.begin(), o2.end());
            }

            return o;
        } else {
            o.emplace_back(nod);
            for(int i=0;i<nod.arity;i++){
                if(nod.child_positions[i][1]==0){
                    std::vector<Node_arma> o2 = active_node(input_nodes[nod.child_positions[i][0]]);
                    o.insert(o.end(), o2.begin(), o2.end());
                }
                else{
                    std::vector<Node_arma> o2 = active_node(node_matrix[nod.child_positions[i][0]][nod.child_positions[i][1] - 1]);
                    o.insert(o.end(), o2.begin(), o2.end());
                }
            }

            return o;
        }
    }

    // Returns active node in graph for each output
    virtual std::vector<Node_arma> get_active_nodes() override{
        std::vector<Node_arma> active_nodes = {};
        for (int i = 0; i < n_outputs; i++) {
            std::vector<Node_arma> o = active_node(output_nodes[i]);
            active_nodes.insert(active_nodes.end(), o.begin(), o.end());
            o.clear();
        }
        return active_nodes;
    }

    // Forward pass through graph for each output
    virtual arma::mat forward(arma::mat x, int batch_size) override{
        std::vector<arma::mat> v;
        for (int i = 0; i < n_outputs; i++) {
            v.emplace_back(node_forward(output_nodes[i], x, batch_size));
        }
        //TODO
        return v[0];
    }

    // Prints nodes using the child positions
    virtual std::string print_active_nodes(bool structural_comparison) override{
        std::string str = "";
        for (int i = 0; i < n_outputs; i++) {
            str += print_node(output_nodes[i], structural_comparison);
        }
        return str;
    }

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

    // The following mutations are used for classic CGP:

    // Mutates the output node
    bool mutate_output(Node_arma selected_node) {
        int idx = selected_node.position[0];
        int r2 = std::rand() % (n_inputs + n_constants + std::min(columns, levels_back) * rows);

        // TODO: bias towards mutating most common node type?
        if (r2 < n_inputs + n_constants) {
            std::vector<int> posi = {r2, 0};

            // If not the same node is picked then the mutation was succesful
            if (posi[0] != output_nodes[idx].child_positions[0][0] ||
                posi[1] != output_nodes[idx].child_positions[0][1]) {
                output_nodes[idx].child_positions[0] = posi;
                return true;
            }
        } else {
            r2 = (r2 - n_inputs - n_constants) + (columns - std::min(columns, levels_back)) * rows;

            std::vector<int> posi = {int(r2 % rows), 1 + int(std::floor(r2 / rows))};

            if (posi[0] != output_nodes[idx].child_positions[0][0] ||
                posi[1] != output_nodes[idx].child_positions[0][1]) {
                output_nodes[idx].child_positions[0] = posi;
                return true;
            }
        }
        return false;
    }

    // Mutates function nodes
    bool mutate_function(Node_arma selected_node) {
        std::vector<int> posi = selected_node.position;

        // 1/3 odds to mutate operator
        if (std::rand() % 3 == 0) {
            node_matrix[posi[0]][posi[1] - 1].mutate_operator(use_indices);
            return true;
        } else {
            int r2 = std::rand() % (n_inputs + n_constants + std::min(posi[1] - 1, levels_back) * rows);

            // Point to new input node
            if (r2 < n_inputs + n_constants) {
                int rc = std::rand() % (node_matrix[posi[0]][posi[1] - 1].child_positions.size());

                // Check whether pointing to different node than before
                if (this->node_matrix[posi[0]][posi[1] - 1].child_positions[rc] != this->input_nodes[r2].position) {
                    this->node_matrix[posi[0]][posi[1] - 1].child_positions[rc] = this->input_nodes[r2].position;

                    return true;
                }
            } else {
                r2 = (r2 - (n_inputs + n_constants)) +
                     ((posi[1] - 1) - std::min((posi[1] - 1), levels_back)) * rows;

                // Choose child
                int rc = std::rand() % (node_matrix[posi[0]][posi[1] - 1].child_positions.size());

                // Choose new position to point to
                std::vector<int> cmp = {int(r2 % rows), 1 + int(std::floor(float(r2) / float(rows)))};
                if (node_matrix[posi[0]][posi[1] - 1].child_positions[rc] != cmp) {
                    node_matrix[posi[0]][posi[1] - 1].child_positions[rc] = cmp;
                    return true;
                }
            }
        }

        return false;
    }

    // Keep mutating until a change to the active graph is made
    virtual bool mutate(string active_nodes) override{
        bool active_node_mutated;

        std::vector<Node_arma> all_nodes;

        for (int output = 0; output < n_outputs; output++) {
            all_nodes.emplace_back(output_nodes[output]);
        }

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < columns; col++) {
                all_nodes.emplace_back(node_matrix[row][col]);
            }
        }

        int r = std::rand() % all_nodes.size();

        Node_arma selected_node = all_nodes[r];

        bool succesfully_mutated = false;

        if (selected_node.is_output) {
            succesfully_mutated = mutate_output(selected_node);
        } else {
            succesfully_mutated = mutate_function(selected_node);
        }

        if (!succesfully_mutated) {
            return false;
        }

        // Check whether expression has changed
        if (active_nodes!=print_active_nodes(false)) {
            active_node_mutated = true;
        } else {
            active_node_mutated = false;
        }

        all_nodes.clear();
        active_nodes.clear();

        return active_node_mutated;
    }

    virtual void initialise_individual(int type) override{
        // Initialise input nodes
        input_nodes.clear();
        for (int i = 0; i < CGP_arma::n_inputs; i++) {
            std::vector<int> pos = {i, 0};
            input_nodes.emplace_back(0, pos, max_erc_value, use_indices);
        }

        // Ensure no constants are initialised
        if(!use_consts){
            CGP_arma::n_constants = 0;
        }

        // Initialise constant nodes
        for (int i = 0; i < CGP_arma::n_constants; i++) {
            std::vector<int> pos = {i + n_inputs, 0};
            input_nodes.emplace_back(1, pos, max_erc_value, use_indices);
        }

        // Initialise function nodes
        node_matrix.clear();

        for (int i = 0; i < CGP_arma::rows; i++) {
            node_matrix.emplace_back(std::vector<Node_arma>({}));
            for (int j = 0; j < CGP_arma::columns; j++) {
                // Offset columns by 1 because 0th column is reserved for inputs and constants
                std::vector<int> pos = {i, j + 1};
                node_matrix[i].emplace_back(3, pos, max_erc_value, use_indices);
            }
        }

        // Type 0 = grow
        for (int j = 0; j < CGP_arma::columns; j++) {
            for (int i = 0; i < CGP_arma::rows; i++) {
                for (int k = 0; k < CGP_arma::max_arity; k++) {
                    if(j==0 || type==0) {
                        // Uniformly choose any node at a preceding level
                        int r = std::rand() % (n_inputs + n_constants + (std::min(j, levels_back) * rows));

                        // If r points to input or constant node
                        //TODO: constants sampled at much higher rate than with GP
                        if (r < n_inputs + n_constants) {
                            node_matrix[i][j].child_positions.emplace_back(input_nodes[r].position);
                            input_nodes[r].parent_positions.emplace_back(node_matrix[i][j].position);
                        } else {
                            // Subtract inputs and constants and add offset
                            r = ((r - n_inputs - n_constants) + (j - std::min(j, levels_back)) * rows);
                            node_matrix[i][j].child_positions.emplace_back(
                                    node_matrix[int(r % rows)][int(std::floor(r / rows))].position);
                            node_matrix[int(r % rows)][int(std::floor(r / rows))].parent_positions.emplace_back(node_matrix[i][j].position);
                        }
                    }
                    else{
                        //TODO: sampling correct? seems as if this is not true full method. Rather grow with the exception of constants and inputs
                        int r = std::rand() % (std::min(j, levels_back) * rows);

                        r = (r + (j - std::min(j, levels_back)) * rows);
                        node_matrix[i][j].child_positions.emplace_back(
                                node_matrix[int(r % rows)][int(std::floor(r / rows))].position);

                        node_matrix[int(r % rows)][int(std::floor(r / rows))].parent_positions.emplace_back(node_matrix[i][j].position);

                    }
                }
            }
        }

        output_nodes.clear();

        for (int i = 0; i < CGP_arma::n_outputs; i++) {
            std::vector<int> pos = {i, columns + 1};
            output_nodes.emplace_back(2, pos, max_erc_value, use_indices);
            if(type==0) {
                //TODO: constants sampled at much higher rate than with GP
                int r = std::rand() % (n_inputs + n_constants + std::min(columns, levels_back) * rows);
                if (r < n_inputs + n_constants) {
                    output_nodes[i].child_positions.emplace_back(input_nodes[r].position);
                    input_nodes[r].parent_positions.push_back(output_nodes[i].position);
                } else {

                    r = (r - n_inputs - n_constants) + (columns - std::min(columns, levels_back)) * rows;
                    // Child of output_nodes does not have the same address as node_matrix;
                    output_nodes[i].child_positions.emplace_back(
                            node_matrix[int(r % rows)][int(std::floor(r / rows))].position);
                    node_matrix[int(r % rows)][int(std::floor(r / rows))].parent_positions.emplace_back(output_nodes[i].position);
                }
            }
            else{
                //TODO: sampling correct? seems as if this is not true full method. Rather grow with the exception of constants and inputs
                int r = std::rand() % (std::min(columns, levels_back) * rows);
                r = r + (columns - std::min(columns, levels_back)) * rows;
                // Child of output_nodes does not have the same address as node_matrix;
                output_nodes[i].child_positions.emplace_back(
                        node_matrix[int(r % rows)][int(std::floor(r / rows))].position);
                node_matrix[int(r % rows)][int(std::floor(r / rows))].parent_positions.emplace_back(output_nodes[i].position);
            }
        }
    }



};

#endif //DCGP_CGP_ARMA_H