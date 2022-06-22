#include <torch/torch.h>

#include "Population/population_arma.h"
#include<fstream>
#include "Tests/tests.h"
#include "Utils/csv_utils.h"
#include <boost/any.hpp>
#include <filesystem>
#include "Experiments/classic_arma.h"

#include "Utils/general_utils.h"
#include "Individual/cgp_arma.h"
#include "Experiments/time_experiment.h"
#include "Experiments/small_tree.h"


namespace fs = filesystem;
using namespace torch;

// Checks whether cuda is available, sets the number of cores available to openmp, sets the seed, prints the current directory.
void prepare_run(int seed){
    // Cuda currently not used
    cout<<"Cuda available: "<< torch::cuda::is_available() << endl;

    int max_threads = omp_get_max_threads();
    cout<<max_threads<<endl;
    int number_of_threads = int(float(max_threads)/1.5);
    omp_set_num_threads(10);
    cout<<"Using: "<< number_of_threads << "/" << max_threads << " threads" << endl;

    cout << "Current path: " << fs::current_path() << endl;

    if(seed==-1){
        seed = time(0);
    }

    srand(seed);
    manual_seed(seed);
    torch::cuda::manual_seed_all(seed);

    cout<<"Using seed: "<<seed<<endl;
}

static void final_evaluation_arma(string filename, Population_arma pop, vector<arma::mat> splits, string const experiment, bool csv, bool linear_after){
    pop.test_mode(true);
    pop.sort_population();

    cout<<"Top individual on training set:"<<endl;
    cout<<pop.pop[0]->print_active_nodes(false)<<endl;

    // Normalise inputs using mean and std found for training split
    if(pop.use_normalise){
        splits[0] = (splits[0].each_row() - pop.mean_normalise).each_row()/pop.std_normalise;
        splits[2] = (splits[2].each_row() - pop.mean_normalise).each_row()/pop.std_normalise;
        splits[4] = (splits[4].each_row() - pop.mean_normalise).each_row()/pop.std_normalise;
        splits[6] = (splits[6].each_row() - pop.mean_normalise).each_row()/pop.std_normalise;
    }

    // Get normalisation and scaling constants from training
    //vector<arma::mat> norm = pop.norm;
    vector<float> a_b = pop.pop[0]->get_ab(splits[0], splits[1]);

    //cout<<"Normalisation variables: "<< norm[0].item<float>()<<", "<<norm[1].item<float>()<< endl;
    cout<<"Scaling : "<< a_b[0]<<", "<<a_b[1]<< endl;

    // Set linear evaluation to true after training
    if(linear_after){
        pop.pop[0]->set_linear(true);
    }

    string res = "";

    cout<<"Test set"<<endl;

    arma::mat test_var_mat = arma::var(splits[5]);
    float test_var = test_var_mat.at(0);
    cout<<"Test variance: "<<test_var<<endl;

    pop.pop[0]->evaluate(splits[4], splits[5]);
    cout<<"MSE: ";
    cout<<pop.pop[0]->get_fitness()<<endl;
    res+= to_string(pop.pop[0]->get_fitness());

    cout<<"NMSE: ";
    cout<<100 * pop.pop[0]->get_fitness()/test_var<<endl;
    res+= "\t" + to_string(pop.pop[0]->get_fitness()/test_var);

    cout << "R2: ";
    cout<<1.-pop.pop[0]->get_fitness()/test_var<<endl;
    res+= "\t" + to_string(1.-pop.pop[0]->get_fitness()/test_var);

    if(splits[2].size()>0) {
        cout << "Validation set" << endl;

        arma::mat validation_var_mat = arma::var(splits[3]);
        float validation_var = validation_var_mat.at(0);
        cout << "Validation variance: " << validation_var << endl;

        pop.pop[0]->evaluate(splits[2], splits[3]);
        cout << "MSE: ";
        cout << pop.pop[0]->get_fitness() << endl;
        res += "\t" + to_string(pop.pop[0]->get_fitness());

        cout << "NMSE: ";
        cout << 100 * pop.pop[0]->get_fitness() / validation_var << endl;
        res += "\t" + to_string(pop.pop[0]->get_fitness() / validation_var);

        cout << "R2: ";
        cout<<1.-pop.pop[0]->get_fitness()/validation_var<<endl;
        res+= "\t" + to_string(1.-pop.pop[0]->get_fitness()/validation_var);
    }
    pop.test_mode(false);

    cout<<"Training set"<<endl;

    arma::mat training_var_mat = arma::var(splits[1]);
    float training_var = training_var_mat.at(0);
    cout<<"Training variance: "<<training_var<<endl;

    pop.pop[0]->evaluate(splits[0], splits[1]);
    cout<<"MSE: ";
    cout<<pop.pop[0]->get_fitness()<<endl;
    res+= "\t" + to_string(pop.pop[0]->get_fitness());

    cout<<"NMSE: ";
    cout<<100 * pop.pop[0]->get_fitness()/training_var<<endl;
    res+= "\t" + to_string(pop.pop[0]->get_fitness()/training_var);

    cout << "R2: ";
    cout<<1.-pop.pop[0]->get_fitness()/training_var<<endl;
    res+= "\t" + to_string(1.-pop.pop[0]->get_fitness()/training_var);

    pop.test_mode(true);

    cout<<"All"<<endl;

    arma::mat total_var_mat = arma::var(splits[7]);
    float total_var = total_var_mat.at(0);
    cout<<"Total variance: "<<total_var<<endl;

    pop.pop[0]->evaluate(splits[6], splits[7]);
    cout<<"MSE: ";
    cout<<pop.pop[0]->get_fitness()<<endl;
    res+= "\t" + to_string(pop.pop[0]->get_fitness());

    cout<<"NMSE: ";
    cout<<100 * pop.pop[0]->get_fitness()/total_var<<endl;
    res+= "\t" + to_string(pop.pop[0]->get_fitness()/total_var);

    cout << "R2: ";
    cout<<1.-pop.pop[0]->get_fitness()/total_var<<endl;
    res+= "\t" + to_string(1.-pop.pop[0]->get_fitness()/total_var);

    cout<<"Number of nodes total: ";
    cout<<pop.pop[0]->get_node_number(false)<<endl;
    res+= "\t" + to_string(pop.pop[0]->get_node_number(false)[0][0]) + "\t" + to_string(pop.pop[0]->get_node_number(false)[0][1]) + "\t" + to_string(pop.pop[0]->get_node_number(false)[0][2]);

    cout<<"Number of nodes unique: ";
    cout<<pop.pop[0]->get_node_number(true)<<endl;
    res+= "\t" + to_string(pop.pop[0]->get_node_number(true)[0][0]) + "\t" + to_string(pop.pop[0]->get_node_number(true)[0][1]) + "\t" + to_string(pop.pop[0]->get_node_number(true)[0][2]);

    res+= "\t" + pop.pop[0]->print_active_nodes(false);

    res += string("\tfitnesses over time\t");

    for(auto fitness:pop.fitness_over_time){
        res += to_string(fitness);
        res +=  ",";
    }
    res = res.substr(0, res.size()-1);

    res += string("\tnode evaluations over time\t");

    for(auto node_eval:pop.node_evals_over_time){
        res += to_string(node_eval);
        res +=  ",";
    }
    res = res.substr(0, res.size()-1);

    if(csv){
        if(filename.empty()){
            write_csv("../data/armatest.csv", experiment + "\t" + res + "\n");
        }
        else{
            write_csv(filename, experiment + "\t" + res + "\n");
        }
    }
}


int main() {
    // Set seed to -1 for random seed
    int seed = -1;
    prepare_run(seed);

    run_classic_experiments_arma_time();
    run_classic_experiments_arma_small_tree();

    return EXIT_SUCCESS;

    // string filename = "../data/datasets/boston.csv";
    // vector<arma::mat> splits = load_tensors_arma(filename, 0.75, 0.);

    // arma::mat tensorX = splits[0];
    // arma::mat tensory = splits[1];

    // cout<<tensorX.size()<<endl;
    // cout<<tensory.size()<<endl;

    // vector<float> max_erc = get_ERC_vals_arma(tensorX);

    // // Run parameters
    // int population_size = 1000; // Population size
    // int generations = 200000; // Total number of generations
    // float time_limit = -1; // Time limit in seconds. Set to -1 to turn off
    // int node_evaluation_limit = -1; // Max number of node_evaluations. Set to -1 to turn off
    // int evaluation_limit = 500000;

    // int n_inputs = get_columns(filename); // Number of inputs

    // int n_constants = 8; // Number of constants (CGP)

    // bool use_consts = false;
    // int n_outputs = 1; // Number of outputs

    // int rows = 8; // Number of rows in node matrix
    // int columns = 3; // Number of columns in node matrix, in tree gp mode this determines the height of the tree

    // int levels_back = 4; // Maximum number of levels back

    // int max_arity = 2; // Maximum arity of operators
    // int max_graph_length = 16; // Max number of operators, constants and inputs in the active graph

    // float percentage_grow = 1.0; // Percentage of individuals initialised with grow. 1.0 for all grow
    // float genitor_percentage = 0.;

    // bool use_linear = true; // Use linear scaling
    // bool use_normalise = false; // Use z-score normalisation

    // vector<float> maxima_erc_values = {-1.,1.}; // Range erc values during initialisation
    // maxima_erc_values = max_erc;

    // int batch_size = 4 ; // Batch size during optimisation

    // int tournament_size = 4; // Tournament size for CGP

    // bool use_tree = false; // Mode where tree is used
    // bool use_gom = true; // Use gene optimal mixing instead of lambda mu

    // bool normalise_MI = true;
    // bool apply_MI_adjustments = true; // Apply Virgolin beta correction on Mutual Information
    // int fos_type = 1; // 0 random tree, 1 LT, 2 univariate FOS

    // int fos_truncation = -1; // Truncate the length of the FOS, -1 uses whole FOS
    // int max_constants = 100; // max bin size
    // float penalty = 99999.; // penalise fitness individual for errors (e.g. division by zero, nans, infs)

    // vector<int> use_indices = {0,1,2,8,9,10,13,17,7,5,6}; // indices of functions to be used

    // bool mix_ERCs = true;

    // Population_arma pop = Population_arma(population_size, generations, time_limit, node_evaluation_limit, evaluation_limit,
    //         n_inputs, n_constants, use_consts, n_outputs,
    //         rows, columns, levels_back,
    //         max_arity, max_graph_length, use_tree, percentage_grow,
    //         use_linear, use_normalise,
    //         maxima_erc_values, batch_size,
    //         tournament_size,
    //         use_gom, fos_type, apply_MI_adjustments, fos_truncation, penalty,
    //         use_indices,  max_constants, seed, genitor_percentage, normalise_MI, mix_ERCs);

    // pop.initialise_data(tensorX, tensory);
    // pop.initialise_population();

    // return EXIT_SUCCESS;
}
