//
// Created by joe on 15-03-21.
//

#include "population_arma.h"
#include <algorithm>
#include <armadillo>
#include <iostream>

#include <chrono>
#include "../GOMEA/gom_arma.h"
#include "../Individual/cgp_arma.h"
#include "../Individual/tree_arma.h"
#include <cassert>
#include <malloc.h>


void Population_arma::del_pop(){
    cout<<"Deleting pop"<<endl;
    for(auto ind:pop){
        delete ind;
    }

    delete pop_gom;
    delete pop_fos;
    malloc_trim(0);
}

Population_arma::Population_arma(int population_size, int generations, float time_limit, int node_evaluation_limit, int evaluation_limit,
                                 int n_inputs, int n_constants, bool use_consts, int n_outputs,
                                 int rows, int columns, int levels_back,
                                 int max_arity, int max_graph_length, bool use_tree, float percentage_grow,
                                 bool use_linear, bool use_normalise,
                                 vector<float> maxima_erc_values, int batch_size,
                                 int tournament_size,
                                 bool use_gom, int fos_type, bool apply_MI_adjustments,
                                 int fos_truncation, float penalty, vector<int> use_indices, int max_constants, int seed,
                                 float genitor_percentage, bool normalise_MI, bool mix_ERCs) {

    Population_arma::population_size = population_size;
    Population_arma::generations = generations;
    Population_arma::time_limit = time_limit;
    Population_arma::node_evaluation_limit = node_evaluation_limit;
    Population_arma::evaluation_limit = evaluation_limit;

    Population_arma::n_inputs = n_inputs;
    Population_arma::n_constants = n_constants;
    Population_arma::use_consts = use_consts;
    Population_arma::n_outputs = n_outputs;

    Population_arma::rows = rows;
    Population_arma::columns = columns;
    Population_arma::levels_back = levels_back;

    Population_arma::max_arity = max_arity;
    Population_arma::max_graph_length = max_graph_length;
    Population_arma::use_tree = use_tree;
    Population_arma::percentage_grow = percentage_grow;
    Population_arma::genitor_percentage = genitor_percentage;

    Population_arma::use_linear = use_linear;
    Population_arma::use_normalise = use_normalise;

    Population_arma::maxima_erc_values = maxima_erc_values;
    Population_arma::batch_size = batch_size;

    Population_arma::tournament_size = tournament_size;

    Population_arma::use_gom = use_gom;
    Population_arma::fos_type = fos_type;
    Population_arma::fos_truncation = fos_truncation;
    Population_arma::normalise_MI = normalise_MI;

    Population_arma::max_constants = max_constants;
    Population_arma::apply_MI_adjustments = apply_MI_adjustments;

    Population_arma::pop = {};
    Population_arma::generation = 0;

    Population_arma::norm = {0,1};

    Population_arma::penalty = penalty;
    // Previous highest fitness
    Population_arma::prev = 9999999999.;
    Population_arma::eps = 0.000001;

    Population_arma::use_indices = use_indices;

    Population_arma::pop_fos = new Fos_arma(apply_MI_adjustments, max_constants, use_tree);
    Population_arma::pop_gom = new gom_arma(false, max_constants, use_tree);

    Population_arma::seed = seed;

    Population_arma::current_mean = -1;
    Population_arma::mean_equal = 0;
    Population_arma::elite_equal = 0;

    Population_arma::mix_ERCs = mix_ERCs;

    if(seed==0){
        cout<<"Warning: thread seeds are all zero due to multiplication by zero seed!"<<endl;
    }

    Population_arma::time = chrono::system_clock::now();
}

// In test mode precomputed linear scaling intercept and slope are used from the training set
void Population_arma::test_mode(bool test_mode_setting){
    for (int i = 0; i < Population_arma::population_size; i++) {
        pop[i]->set_test_mode(test_mode_setting);
    }
}

// Initialise input and target tensors
void Population_arma::initialise_data(arma::mat initX, const arma::mat inity) {
    Population_arma::X = initX;
    Population_arma::y = inity;

    // Calculate mean and std of training data input and response and normalise training data
    if(use_normalise){
        Population_arma::mean_normalise = arma::mean(initX, 0);
        Population_arma::std_normalise = arma::stddev(initX, 0);
        Population_arma::X = (initX.each_row()-mean_normalise).each_row()/std_normalise;

        vector<float> res;

        arma::mat meany = arma::mean(inity);
        res.emplace_back(meany.at(0));
        arma::mat stdy = arma::stddev(inity);
        res.emplace_back(stdy.at(0));

        Population_arma::norm = res;
    }
}

struct PointerCompare {
    bool operator()(Individual_arma* l, Individual_arma* r) {
        return l->get_fitness() < r->get_fitness();
    }
};

// Sort Population_arma on fitness
void Population_arma::sort_population() {
    sort(pop.begin(), pop.end(), PointerCompare());
}

Individual_arma* Population_arma::add_Individual_arma_to_pop(int init_type){
    Individual_arma *ind;
    if(use_tree){

        ind = new Tree_arma(n_inputs, n_constants, use_consts, n_outputs, rows, columns, levels_back,
                            max_arity, batch_size, use_linear, maxima_erc_values, init_type,
                            use_normalise, max_graph_length, penalty,
                            &use_indices);
        ind->change_norm(norm);

    }
    else{
        ind = new CGP_arma(n_inputs, n_constants, use_consts, n_outputs, rows, columns, levels_back,
                           max_arity, batch_size, use_linear, maxima_erc_values, init_type, use_normalise,
                            max_graph_length, penalty, &use_indices);

    }
    return ind;
}

// Initialise Population_arma with Individual_armas
void Population_arma::initialise_population() {
    pop.clear();

    for (int i = 0; i < Population_arma::population_size; i++) {
        Individual_arma* ind;
        if(i>=int(Population_arma::population_size*percentage_grow)){
            ind = add_Individual_arma_to_pop(1);
        }
        else{
            ind = add_Individual_arma_to_pop(0);
        }
        pop.emplace_back(ind);
        pop[i]->change_norm(norm);
    }
}


void Population_arma::count_evalutions(){
    //#pragma omp parallel for reduction(+:total_evaluations) schedule(static) default(none)
    for (int i = 0; i < population_size; i++) {
        total_evaluations += pop[i]->get_evaluations();
        total_node_evaluations += pop[i]->get_node_evaluations();
        pop[i]->clear_evaluations();
        pop[i]->clear_node_evaluations();
    }
}

// Evaluate Population_arma/update fitness
void Population_arma::evaluate_population() {
//    int total_node_evaluations = 0;
//    vector<int> reasons = {0,0,0,0,0,0,0};
// shared(reasons)
    //#pragma omp parallel for reduction(+:total_node_evaluations) schedule(static) default(none)
    for (int i = 0; i < population_size; i++) {
        int reason = pop[i]->evaluate(X, y);
//        reasons[reason]++;
        total_node_evaluations += pop[i]->get_node_evaluations();
    }
//    cout<<"REASONS"<<endl;
//    for(auto reason:reasons){
//        cout<<reason<<" ";
//    }
    cout<<endl;
}

// Check whether a set time limit is reached
const bool Population_arma::time_limit_reached() {
    if (time_limit > 0.0) {
        if (chrono::system_clock::now() - time > chrono::duration<float>(time_limit)) {
            return true;
        }
    }
    return false;
}

// Tournament selection
Individual_arma* Population_arma::tournament_selection() {
    int r = rand() % population_size;
    Individual_arma* ind1 = pop[r]->clone();
    for (int i = 0; i < tournament_size; i++) {
        r = rand() % population_size;
        if (ind1->get_fitness() > pop[r]->get_fitness()) {
            delete ind1;
            ind1 = pop[r]->clone();
        }
    }
    return ind1;
}

float Population_arma::get_mean_fitness() {
    float sum = 0.0;
    //#pragma omp parallel for shared(Population_arma::pop) reduction(+:sum) default(none)
    for (int i = 0; i < population_size; i++) {
        sum += pop[i]->get_fitness();
    }
    sum /= float(population_size);
    return sum;
}

// Originally the parent is the Individual_arma with the highest fitness. The offspring is a mutated version of the parent.
// I've changed it slightly, now tournament selection is used to select a parent.
// Elitism = 1
vector<Individual_arma*> Population_arma::lambda_mu() {
    vector<Individual_arma*> new_Population_arma;
    new_Population_arma.resize(population_size);
    new_Population_arma[0] = pop[0]->clone();

    //#pragma omp parallel for shared(Population_arma::pop)
    for (int i = 1; i < population_size; i++) {
        Individual_arma* clone = tournament_selection();
        clone->mutate_until_active();
        new_Population_arma[i] = clone;
        new_Population_arma[i]->evaluate(X, y);
    }

    for(int i = 0; i<population_size; i++){
        delete pop[i];
    }

    return new_Population_arma;
}

// Report stats
bool Population_arma::report_stats() {
    float mean = get_mean_fitness();

    if(abs(mean - current_mean)<eps){
        mean_equal += 1;
    }
    else{
        mean_equal = 0;
    }

    if(fitness_over_time.size()>0  && abs(fitness_over_time[fitness_over_time.size() - 1] - Population_arma::pop[0]->get_fitness())<eps){
        elite_equal += 1;
    }
    else{
        elite_equal = 0;
    }



    current_mean = mean;

    cout<<endl;
    cout << "Generation: " << generation << " Top Fitness: " << Population_arma::pop[0]->get_fitness() << " Mean Fitness: "<< get_mean_fitness() << " time elapsed (s): "
         << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - time).count()/1000.<<std::endl;


    cout<<Population_arma::pop[0]->print_active_nodes(false)<<endl;

    fitness_over_time.push_back(Population_arma::pop[0]->get_fitness());

    times.push_back(chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - time).count()/1000.);

    bool converged = false;

    if(abs(mean - Population_arma::pop[0]->get_fitness())<eps || Population_arma::pop[0]->get_fitness() > mean){
        cout<<"Converged: mean and elite equal."<<endl;
        converged = true;
    }
    else if(mean_equal==5){
        cout<<"Converged: mean fitness equal for 5 generations"<<endl;
        converged = true;
    }
    else if(elite_equal==100){
        cout<<"Converged: elite fitness equal for 100 generations"<<endl;
        converged = true;
    }

    return  converged;
}

// Apply Gene optimal mixing to entire Population_arma
vector<Individual_arma*> Population_arma::apply_fast_gom(){
    auto t = chrono::system_clock::now();

    vector<Individual_arma*> new_Population_arma;
    new_Population_arma.resize(population_size);

    //#pragma omp parallel for schedule(static)
    for(int i = 0; i<population_size; i++){
        new_Population_arma[i] = Population_arma::pop_gom->fast_mix(i, pop_fos->subsets, seed*(i+1)*(this->generation+1), &X, &y, pop, &rows, &columns, &n_outputs, &fos_truncation);
    }

    for(int i = 0; i<population_size; i++){


        delete pop[i];
    }


    cout<<"GOM MIXING: "<<chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - t).count()/1000.<<endl;
    return new_Population_arma;
}

void Population_arma::fos(){
    pop_fos->build_fos(fos_type, rows, columns, n_outputs, pop, &MI_adjustments, use_indices.size(), n_inputs, normalise_MI, use_indices, mix_ERCs);
}

// Evolve Population_arma
// 1. Evaluate
// 2. Sort on fitness
// 3. Optimise
// 4. Build FOS (if using GOM)
// 5. Alter Population_arma
void Population_arma::evolve() {

//    std::vector<std::vector<std::vector<float>>> M = {};
//    string line;
//    ifstream f ("MIs.txt");
//
//    int curr_row = 0;
//    std::vector<std::vector<float>> block = {};
//    if (f.is_open()) {
//        while (getline(f, line)) {
//            if(line=="-"){
//                M.emplace_back(block);
//                block = {};
//            }
//            else{
//                std::vector<float> line_float = {};
//                std::stringstream ss(line);
//                float val;
//                while(ss >> val){
//                    line_float.emplace_back(val);
//
//                    if(ss.peek() == ',') ss.ignore();
//
//                }
//                block.emplace_back(line_float);
//            }
//        }
//    }

    //Initial evaluation
    evaluate_population();

    for (generation = 0; generation < generations; generation++) {
        count_evalutions();

        cout<<"Total Node evaluations: "<<total_node_evaluations<<" Total Evaluations: "<<total_evaluations<<endl;
        node_evals_over_time.push_back(total_node_evaluations);

        sort_population();

        // Add new individuals
        if(genitor_percentage>0.0){
            for(int x = population_size-1; x>population_size - int(population_size*genitor_percentage); x--){
                delete pop[x];
                //TODO should be according to grow_percentage
                if(rand()%2==0){
                    pop[x] = add_Individual_arma_to_pop(0);
                }
                else{
                    pop[x] = add_Individual_arma_to_pop(1);
                }

                pop[x]->evaluate(X, y);
            }
        }

        // Time limit check
        if (time_limit_reached()) {
            cout << "Time limit reached" << endl;
            break;
        }


        // Convergence check on elite. If elite has a fitness very close to zero the Population_arma is deemed to have converged to the correct formula
        if (Population_arma::pop[0]->get_fitness() <= eps){
            cout<<Population_arma::pop[0]->get_fitness()<<endl;
            Population_arma::pop[0]->evaluate(X, y);
            cout<<Population_arma::pop[0]->get_fitness()<<endl;
            cout << "Converged: elite close to correct formula" << endl;
            break;
        }

        if(use_gom){
            auto t = chrono::system_clock::now();
            pop_fos->build_fos(fos_type, rows, columns, n_outputs, pop, &MI_adjustments, use_indices.size(), n_inputs, normalise_MI, use_indices, mix_ERCs);

//            cout<<"WARNING MI REPLACEMENT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
//            pop_fos->replace_MI(M[generation], (pow(2, columns) - 1) / (1));
            cout<<"FOS BUILD "<<chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - t).count()/1000.<<endl;
        }

        // In all paradigms the elites fitness should never worsen. Small numerical instabilities are taken into account.

        if(Population_arma::pop[0]->get_fitness() >= prev && (Population_arma::pop[0]->get_fitness()-prev)>eps){
            cout<<"top fitness: "<<Population_arma::pop[0]->get_fitness()<<" prev: "<<prev<<endl;
            throw runtime_error("Fitness worsened!");
        }
        prev = Population_arma::pop[0]->get_fitness();

        bool converged = report_stats();

        if(node_evaluation_limit>-1 && total_node_evaluations>=node_evaluation_limit){
            cout << "Node evaluation limit reached" << endl;
            break;
        }

        if(evaluation_limit>-1 && total_evaluations>=evaluation_limit){
            cout << "Evaluation limit reached" << endl;
            break;
        }

        // Another convergence check. If the mean fitness is equal to the top fitness then the population has converged
        if(converged){
            cout << "Converged: mean and top fitness equal" << endl;
            break;
        }
        if (use_gom) {
            pop = apply_fast_gom();

        } else {
            pop = lambda_mu();
        }

    }

    evaluate_population();
    sort_population();

    cout<<"fitness over time:"<<endl;
    for(auto f:fitness_over_time){
        cout<<f<<", ";
    }
    cout<<endl;

    cout<<"node evals over time:"<<endl;
    for(auto f:node_evals_over_time){
        cout<<f<<", ";
    }
    cout<<endl;

    cout<<"time over time:"<<endl;
    for(auto f:times){
        cout<<f<<", ";
    }
    cout<<endl;

    cout << "Max generations reached" << endl;
}