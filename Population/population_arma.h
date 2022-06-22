//
// Created by joe on 07-10-21.
//

//
// Created by joe on 15-03-21.
//

#ifndef DCGP_Population_arma_H
#define DCGP_Population_arma_H

#include <vector>
#include "../Individual/individual_arma.h"
#include "../GOMEA/fos_arma.h"
#include "../GOMEA/gom_arma.h"
#include <armadillo>


class Population_arma {

public:
    void del_pop();

    Population_arma(int population_size, int generations, float time_limit, int node_evaluation_limit, int evaluation_limit,
                    int n_inputs, int n_constants, bool use_consts, int n_outputs,
                    int rows, int columns, int levels_back,
                    int max_arity, int max_graph_length, bool use_tree, float percentage_grow,
                    bool use_linear, bool use_normalise,
                    std::vector<float> maxima_erc_values, int batch_size,
                    int tournament_size,
                    bool use_gom,
                    int fos_type, bool apply_MI_adjustments, int fos_truncation, float penalty,
                    std::vector<int> use_indices, int max_constants, int seed, float genitor_percentage, bool normalise_MI, bool mix_ERCs);

    int population_size;
    int generations;
    float time_limit;

    int node_evaluation_limit;
    int evaluation_limit;

    int n_inputs;
    int n_constants;
    bool use_consts;
    int n_outputs;

    int rows;
    int columns;
    int levels_back;

    int max_arity;
    int max_graph_length;
    bool use_tree;
    float percentage_grow;
    float genitor_percentage;

    bool use_linear;
    bool use_normalise;

    std::vector<float> maxima_erc_values;
    int batch_size;

    int tournament_size;

    bool use_gom;

    int fos_type;
    bool apply_MI_adjustments;
    int fos_truncation;
    bool normalise_MI;

    int max_constants;

    float penalty;
    float eps;

    std::vector<int> use_indices;

    arma::mat mean_normalise;
    arma::mat std_normalise;

    std::vector<std::vector<float>> MI_adjustments;

    float prev;
    Fos_arma* pop_fos;
    gom_arma* pop_gom;

    std::vector<float> norm;

    int seed;


    std::vector<float> fitness_over_time;
    std::vector<int> node_evals_over_time;
    std::vector<float> times;

    arma::mat X;
    arma::mat y;

    int total_node_evaluations = 0;
    int total_evaluations = 0;
    int generation;
    std::vector<Individual_arma*> pop;

    std::chrono::time_point<std::chrono::system_clock> time;

    int mean_equal;
    int elite_equal;
    float current_mean;

    bool mix_ERCs;

    void initialise_data(arma::mat X, arma::mat y);

    void test_mode(bool test_mode_setting);

    void sort_population();

    void evaluate_population();

    void evolve();

    const bool time_limit_reached();

    bool report_stats();

    float get_mean_fitness();

    std::vector<Individual_arma*> apply_fast_gom();

    void initialise_population();

    Individual_arma* add_Individual_arma_to_pop(int);

    std::vector<Individual_arma*> lambda_mu();

    Individual_arma* tournament_selection();

    void fos();

    void count_evalutions();
};


#endif //DCGP_Population_arma_H
