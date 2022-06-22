//
// Created by joe on 07-10-21.
//

#ifndef DCGP_CGP_GPMODE_ARMA_H
#define DCGP_CGP_GPMODE_ARMA_H

struct CGP_GPMODE_arma: public Individual_arma {
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

    std::vector<float> norm;

    std::vector<float> max_erc_value;

    float a;
    float b;

    float gradient_clip;
    float inject_noise;

    std::vector<std::string> op_strings;

    std::vector<std::vector<Node_arma>> node_matrix;
    std::vector<Node_arma> output_nodes;

public:
    CGP_GPMODE_arma(int n_inputs, int n_constants, bool use_consts, int n_outputs,
                    int rows, int columns, int levels_back, int max_arity,
                    int batch_size, bool linear, std::vector<float> max_erc_value, int init_type, bool normalise,
                    float gradient_clip, float inject_noise, int max_length, float penalty, std::vector<int> *use_indices){

        CGP_GPMODE_arma::n_inputs = n_inputs;
        CGP_GPMODE_arma::n_constants = n_constants;
        CGP_GPMODE_arma::use_consts = use_consts;
        CGP_GPMODE_arma::n_outputs = n_outputs;

        CGP_GPMODE_arma::rows = rows;
        CGP_GPMODE_arma::columns = columns;
        CGP_GPMODE_arma::levels_back = levels_back;

        CGP_GPMODE_arma::max_arity = max_arity;
        CGP_GPMODE_arma::max_length = max_length;

        CGP_GPMODE_arma::batch_size = batch_size;

        CGP_GPMODE_arma::test_mode = false;

        CGP_GPMODE_arma::fitness = 0.;
        CGP_GPMODE_arma::penalty = penalty;

        CGP_GPMODE_arma::linear = linear;
        CGP_GPMODE_arma::normalise = normalise;
        CGP_GPMODE_arma::max_erc_value = max_erc_value;


        CGP_GPMODE_arma::gradient_clip = gradient_clip;
        CGP_GPMODE_arma::inject_noise = inject_noise;

        CGP_GPMODE_arma::use_indices = use_indices;
        CGP_GPMODE_arma::op_strings = {"add", "multiply", "subtract", "divide", "analytic_quotient","max", "min", "pow", "sin", "cos", "sqrt", "log", "exp", "square"};

        node_matrix = {};
        output_nodes = {};
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
        node_matrix[i][j].is_input = is_input;
        node_matrix[i][j].is_constant = is_constant;
        node_matrix[i][j].is_output = is_output;
    }

    virtual void set_input_number(int i, int j, int input_number){
        node_matrix[i][j].input_number = input_number;
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

    virtual void set_test_mode(bool test_mode) override{
        CGP_GPMODE_arma::test_mode = test_mode;
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

    virtual void set_node_matrix(int i, int j, float t, std::vector<int> pos){

        node_matrix[i][j] = Node_arma(1, pos, max_erc_value, use_indices);
        node_matrix[i][j].value = t;
    }

    virtual bool get_normalise() override{
        return normalise;
    }

    virtual bool get_linear() override{
        return linear;
    }

    virtual float get_gradient_clip() override{
        return gradient_clip;
    }

    virtual float get_inject_noise() override{
        return inject_noise;
    }

    virtual void set_child_positions(int i, int j, std::vector<int> pos){
        node_matrix[i][j].child_positions.emplace_back(pos);
    }

    virtual vector<vector<int>> get_child_positions(int i, int j){
        return node_matrix[i][j].child_positions;
    }

    //Copy constructor. Important note: without copy constructor after mutation the children will point to the wrong nodes!
    virtual Individual_arma* clone() override{

        Individual_arma* new_ind = new CGP_GPMODE_arma(*this);
        return new_ind;
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

    virtual int get_node_evaluations(){
        return node_evaluations;
    }

    // Returns a string of the active nodes. A Node_arma is active when it connected to an output node
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

            if(linear && normalise){
                str += std::string("output_") + std::to_string(nod.position[0]) + std::string("=") +
                       std::to_string(norm[0]) + "+" +  std::to_string(norm[1]) +
                       "*" "(" + std::to_string(a) + "+" + std::to_string(b) + "*" + "(" +
                       print_node(node_matrix[nod.child_positions[0][0]][nod.child_positions[0][1]], structural_comparison) + "))";
            }
            else if(linear){
                str += std::string("output_") + std::to_string(nod.position[0]) + std::string("=") +
                       std::to_string(a) + "+" + std::to_string(b) + "*" +
                       print_node(node_matrix[nod.child_positions[0][0]][nod.child_positions[0][1]], structural_comparison);
            }
            else if(normalise){
                str += std::string("output_") + std::to_string(nod.position[0]) + std::string("=") +
                       std::to_string(norm[0]) + "+" +  std::to_string(norm[1]) +
                       "*" "(" +
                       print_node(node_matrix[nod.child_positions[0][0]][nod.child_positions[0][1]], structural_comparison) + ")";
            }
            else{
                str += std::string("output_") + std::to_string(nod.position[0]) + std::string("=") +
                       print_node(node_matrix[nod.child_positions[0][0]][nod.child_positions[0][1]], structural_comparison);
            }


        } else {
            str += op_strings[nod.op] + std::string("(");
            for (int i = 0; i < nod.arity; i++) {

                //str += print_node(node_matrix[nod.child_positions[i][0]][nod.child_positions[i][1]]);
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

    // TODO: When there is no input in the entire graph the output size is not broadcasted in the correct format?
    // Returns the output tensor of a Node_arma given an input tensor x and a batch size
    arma::mat node_forward(Node_arma nod, arma::mat x, int batch_size, bool use_caching){
        if (nod.is_input) {
            node_evaluations++;
            return x.col(nod.input_number);
        }
        else if (nod.is_constant) {
            node_evaluations++;
            return arma::ones(batch_size, 1) * nod.value;
        }
        else if (nod.is_output) {
            if(!nod.cached_value.empty()) {
                return nod.cached_value;
            }
            arma::mat forward = node_forward(node_matrix[nod.child_positions[0][0]][nod.child_positions[0][1]], x, batch_size, use_caching);
            if(use_caching){
                nod.cached_value = forward;
            }
            node_evaluations++;
            return forward;
        }
        else {
            if(!nod.cached_value.empty()){
                return nod.cached_value;
            }
            std::vector<arma::mat> c;
            for (int i = 0; i < nod.arity; i++) {
                c.emplace_back(node_forward(node_matrix[nod.child_positions[i][0]][nod.child_positions[i][1]], x,
                                            batch_size, use_caching));
            }
            switch (nod.op) {
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
                    x = arma::sign(c[1]) * (c[0] / (arma::abs(c[1]) + 10e-6));
                    break;
                case 5:
                    x = arma::max(c[0], c[1]);
                    break;
                case 6:
                    x = arma::min(c[0], c[1]);
                    break;
                case 7:
                    //TODO
                    //x = arma::pow(c[0], c[1]);
                    break;
                case 8:
                    x = arma::sin(c[0]);
                    break;
                case 9:
                    x = arma::cos(c[0]);
                    break;

                case 10:
                    x = arma::sqrt(c[0]);
                    break;
                case 11:
                    x = arma::log(c[0]);
                    break;
                case 12:
                    x = arma::exp(c[0]);
                    break;
                case 13:
                    x = arma::square(c[0]);
                    break;
            }

            if(use_caching){
                nod.cached_value = x;
            }
            node_evaluations++;
            return x;
        }
    }

    // TODO: Is the active graph correct when the arity is 1?
    // Returns whether a Node_arma is used in the graph by traversing the graph from each output node
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

            std::vector<Node_arma> o2 = active_node(node_matrix[nod.child_positions[0][0]][nod.child_positions[0][1]]);
            o.insert(o.end(), o2.begin(), o2.end());


            return o;
        } else {
            o.emplace_back(nod);
            if(nod.child_positions.size()>0) {
                for (int i = 0; i < nod.arity; i++) {

                    std::vector<Node_arma> o2 = active_node(
                            node_matrix[nod.child_positions[i][0]][nod.child_positions[i][1]]);
                    o.insert(o.end(), o2.begin(), o2.end());

                }
            }

            return o;
        }
    }

    // Returns active Node_arma in graph for each output
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
    virtual arma::mat forward(arma::mat x, int batch_size, bool use_caching) override{
        std::vector<arma::mat> v;
        for (int i = 0; i < n_outputs; i++) {
            v.emplace_back(node_forward(output_nodes[i], x, batch_size, use_caching));
        }

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

    // Changes the normalisation factors, tensor is as follows [intercept, slope]
    virtual void change_norm(std::vector<float> norm) override{
        CGP_GPMODE_arma::norm = norm;
    }

    // Evaluate the fitness of individual. Int that is returned can optionally be used for debuggging
    virtual int evaluate(arma::mat X, arma::mat y, bool use_caching) override{
        fitness = 0;

        //TODO: parameterise penalty
        if(max_length > 0  && get_active_nodes().size()>max_length){
            fitness += penalty;
            return 0;
        }

        // Equivalent to with torch.no_grad(), no gradient information saved


        arma::mat output;
        try {
            output = forward(X, X.n_rows, use_caching);
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

    // TODO: parameterise sigma and mean
    // Mutates a constant Node_arma by adding Gaussian noise
    bool mutate_constant(Node_arma selected_node) {
        //selected_node.value = selected_node.value + torch::randn({1}, arma::matOptions().dtype(torch::kFloat));
        return true;
    }

    // Mutates the output node
    bool mutate_output(Node_arma selected_node) {
        int idx = selected_node.position[0];
        int r2 = std::rand() % (n_inputs + n_constants + std::min(columns, levels_back) * rows);

        // TODO: bias towards mutating most common Node_arma type?
        if (r2 < n_inputs + n_constants) {
            std::vector<int> posi = {r2, 0};

            // If not the same Node_arma is picked then the mutation was succesful
            if (posi[0] != output_nodes[idx].child_positions[0][0] ||
                posi[1] != output_nodes[idx].child_positions[0][1]) {
                output_nodes[idx].child_positions[0] = posi;
                return true;
            }
        } else {
            r2 = r2 - n_inputs - n_constants + (columns - std::min(columns, levels_back)) * rows;

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




        if (std::rand() % 3 == 0) {
            node_matrix[posi[0]][posi[1]].mutate_operator(use_indices);
            return true;
        } else {
            int r2 = std::rand() % (std::min(posi[1], levels_back) * rows);


            // Choose child
            int rc = std::rand() % (node_matrix[posi[0]][posi[1]].child_positions.size());

            // Choose new position to point to
            std::vector<int> cmp = {int(r2 % rows), 1 + int(std::floor(float(r2) / float(rows)))};
            if (node_matrix[posi[0]][posi[1]].child_positions[rc] != cmp) {
                node_matrix[posi[0]][posi[1]].child_positions[rc] = cmp;
                return true;
            }

        }


        return false;
    }

    // Keep mutating untill a change to the active graph is made
    virtual bool mutate(string active_nodes) override{
        bool active_node_mutated;

        int r = std::rand() % (columns * rows + n_outputs);

        // Gather all nodes in graph
        std::vector<Node_arma> all_nodes;

        for (int output = 0; output < n_outputs; output++) {
            all_nodes.emplace_back(output_nodes[output]);
        }

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < columns; col++) {
                all_nodes.emplace_back(node_matrix[row][col]);
            }
        }

        Node_arma selected_node = all_nodes[r];

        int succesfully_mutated = 0;

        if (selected_node.is_output) {
            succesfully_mutated = mutate_output(selected_node);
        } else if (selected_node.is_input) {
            return false;
        } else if (selected_node.is_constant) {
            succesfully_mutated = mutate_constant(selected_node);
        } else {
            succesfully_mutated = mutate_function(selected_node);
        }

        if (!succesfully_mutated) {
            return false;
        }

        if (active_nodes!=print_active_nodes(false)) {
            active_node_mutated = true;
        } else {
            active_node_mutated = false;
        }

        return active_node_mutated;
    }



    virtual void initialise_individual(int type) override{
        // Initialise matrix nodes
        node_matrix.clear();

        for (int i = 0; i < CGP_GPMODE_arma::rows; i++) {
            node_matrix.emplace_back(std::vector<Node_arma>({}));
            for (int j = 0; j < CGP_GPMODE_arma::columns; j++) {
                std::vector<int> pos = {i, j};
                // First column is always a constant or input
                if(j==0){
                    vector<int> choices = {0,1};
                    int r;
                    //if constants are used 50/50 chance of picking a constant for the first column position
                    if(use_consts && n_constants>0){
                        r = std::rand() % 2;
                    }
                    else{
                        r = 0;
                    }

                    node_matrix[i].emplace_back(choices[r], pos, max_erc_value, use_indices);
                    if(r==0){
                        // Because gpmode is used the node's contents are changed and not the connections so an input
                        // number must be specified as the position does not define the input number
                        node_matrix[i][j].set_input_number(int(std::rand() % n_inputs));
                    }
                }
                else{
                    // if Grow type
                    if(type==0){
                        vector<int> choices;
                        if(use_consts){
                            choices = {0, 1, 3};
                        }
                        else{
                            choices = {0, 3};
                        }

                        int r = std::rand() % choices.size();
                        node_matrix[i].emplace_back(choices[r], pos, max_erc_value, use_indices);
                        if(r==0){
                            node_matrix[i][j].set_input_number(int(std::rand() % n_inputs));
                        }
                    }
                    else{
                        node_matrix[i].emplace_back(3, pos, max_erc_value, use_indices);
                    }
                }
            }
        }

        //Add connections
        //Column start at 1 because the first columns contains input and constant nodes
        for (int j = 1; j < CGP_GPMODE_arma::columns; j++) {
            for (int i = 0; i < CGP_GPMODE_arma::rows; i++) {
                for (int k = 0; k < CGP_GPMODE_arma::max_arity; k++) {
                    if(type==0 && j>1){
                        //TODO: add levels back;
                        int c = std::rand() % j;
                        int r = std::rand()%CGP_GPMODE_arma::rows;
                        node_matrix[i][j].child_positions.emplace_back(node_matrix[r][c].position);
                        node_matrix[r][c].parent_positions.emplace_back(node_matrix[i][j].position);
                    }
                    else{
                        int c = j - 1;
                        int r = std::rand()%CGP_GPMODE_arma::rows;
                        node_matrix[i][j].child_positions.emplace_back(node_matrix[r][c].position);
                        node_matrix[r][c].parent_positions.emplace_back(node_matrix[i][j].position);
                    }
                }
            }
        }
        //Connect the output nodes
        output_nodes.clear();

        for (int i = 0; i < CGP_GPMODE_arma::n_outputs; i++) {
            std::vector<int> pos = {i, columns};
            output_nodes.emplace_back(2, pos, max_erc_value, use_indices);
            if(type==0) {
                int c = std::rand()%CGP_GPMODE_arma::columns;
                int r = std::rand()%CGP_GPMODE_arma::rows;
                output_nodes[i].child_positions.emplace_back(node_matrix[r][c].position);
                node_matrix[r][c].parent_positions.emplace_back(output_nodes[i].position);
            }
            else{
                int c = CGP_GPMODE_arma::columns - 1;
                int r = std::rand()%CGP_GPMODE_arma::rows;
                output_nodes[i].child_positions.emplace_back(node_matrix[r][c].position);
                node_matrix[r][c].parent_positions.emplace_back(output_nodes[i].position);
            }
        }
    }

};

#endif //DCGP_CGP_GPMODE_ARMA_H