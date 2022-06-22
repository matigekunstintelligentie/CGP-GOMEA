//
// Created by joe on 12-04-21.
//

#ifndef DCGP_TESTS_H
#define DCGP_TESTS_H

#include "../GOMEA/upgma.h"
#include "../Utils/general_utils.h"
#include "../Utils/csv_utils.h"
#include "../Individual/tree_arma.h"
#include "../Individual/cgp_arma.h"

using namespace std;
using namespace torch;

bool thread_reproducibility_arma_test(){
    int seed = 42;

    int population_size = 10;

    srand(seed);
    manual_seed(seed);
    torch::cuda::manual_seed_all(seed);

    string filename = "../data/datasets/boston.csv";
    vector<arma::mat> splits = load_tensors_arma(filename, 0.5, 0.25);

    arma::mat tensorX = splits[0];
    arma::mat tensory = splits[1];


    Population_arma pop = Population_arma(population_size, 2, 1000, -1, -1,
                                13, 13, false, 1,
                                3, 3, 1,
                                2, 32, false, 0.5,
                                1, 0,
                                {-1., 1.}, 16,
                                5,
                                1, 1, 1, -1, 99999., vector<int>({0,1,2,3}), 1000000, 42, 0.0, true, false);
    pop.initialise_population();
    pop.initialise_data(tensorX, tensory);

    srand(seed);
    manual_seed(seed);
    torch::cuda::manual_seed_all(seed);

    vector<arma::mat> splits2 = load_tensors_arma(filename, 0.5, 0.25);

    arma::mat tensorX2 = splits2[0];
    arma::mat tensory2 = splits2[1];


    //Splits check
    arma::mat diff = arma::abs(arma::sum(tensorX2 - tensorX));
    if(diff.at(0) > 0.0000001){
        cout<<"Inequality in input data"<<endl;
        return false;
    }


    Population_arma pop2 = Population_arma(population_size, 2, 1000, -1, -1,
                                 13, 13, false, 1,
                                 3, 3, 1,
                                 2, 32, false, 0.5,
                                 1, 0,
                                 {-1., 1.}, 16,
                                 5,
                                 1, 1, 1, -1, 99999., vector<int>({0,1,2,3}), 1000000, 42, 0.0, true, false);
    pop2.initialise_population();
    pop2.initialise_data(tensorX2, tensory2);

    for(int i=0;i<population_size;i++){
        if(pop.pop[i]->print_active_nodes(false)!=pop2.pop[i]->print_active_nodes(false)){
            cout<<i<<" inequality during initialisation"<<endl;
            return false;
        }
    }

    srand(seed);
    manual_seed(seed);
    torch::cuda::manual_seed_all(seed);
    pop.evolve();

    vector<string> pop_str = {};
    vector<string> pop_str_cache = {};
    for(int i=0;i<population_size;i++){
        pop_str.push_back(pop.pop[i]->print_active_nodes(false));
        pop_str_cache.push_back(pop.pop[i]->print_active_nodes(true));
    }

    pop.del_pop();

    srand(seed);
    manual_seed(seed);
    torch::cuda::manual_seed_all(seed);
    pop2.evolve();

    for(int i=0;i<population_size;i++){
        if(pop_str[i]!=pop2.pop[i]->print_active_nodes(false)){
            cout<<pop_str[i]<<endl;
            cout<<pop2.pop[i]->print_active_nodes(false)<<endl;
            cout<<pop_str_cache[i]<<endl;
            cout<<pop2.pop[i]->print_active_nodes(true)<<endl;
            cout<<i<<"inequality during evolution"<<endl;
            return false;
        }
    }

    return true;
}



bool cache_equivalence_test(bool use_tree){
    int seed = 42;

    int population_size = 4;
    int generations = 1;

    srand(seed);
    manual_seed(seed);
    torch::cuda::manual_seed_all(seed);

    string filename = "../data/datasets/boston.csv";
    vector<arma::mat> splits = load_tensors_arma(filename, 0.5, 0.25);

    arma::mat tensorX = splits[0];
    arma::mat tensory = splits[1];

    Population_arma pop = Population_arma(population_size, generations, 1000, -1, -1,
                                          13, 13, true, 1,
                                          3, 3, 1,
                                          2, 32, use_tree, 0.5,
                                          1, 0,
                                          {-1., 1.}, 16,
                                          5,
                                          1, 1, 1, -1, 99999., vector<int>({0,1,2,3}), 100, seed, 0.0, true, false);

    pop.initialise_population();
    pop.initialise_data(tensorX, tensory);

    srand(seed);
    manual_seed(seed);
    torch::cuda::manual_seed_all(seed);

    vector<arma::mat> splits2 = load_tensors_arma(filename, 0.5, 0.25);

    arma::mat tensorX2 = splits2[0];
    arma::mat tensory2 = splits2[1];


    //Splits check
    arma::mat diff = arma::abs(arma::sum(tensorX2 - tensorX));
    if(diff.at(0) > 0.0000001){
        cout<<"Inequality in input data"<<endl;
        return false;
    }


    Population_arma pop2 = Population_arma(population_size, generations, 1000, -1, -1,
                                           13, 13, true, 1,
                                           3, 3, 1,
                                           2, 32, use_tree, 0.5,
                                           1, 0,
                                           {-1., 1.}, 16,
                                           5,
                                           1, 1, 1, -1, 99999., vector<int>({0,1,2,3}), 100, seed, 0.0, true, false);
    pop2.initialise_population();
    pop2.initialise_data(tensorX2, tensory2);

    for(int i=0;i<population_size;i++){
        if(pop.pop[i]->print_active_nodes(false)!=pop2.pop[i]->print_active_nodes(false)){
            cout<<i<<" inequality during initialisation"<<endl;
            return false;
        }
    }

    srand(seed);
    manual_seed(seed);
    torch::cuda::manual_seed_all(seed);
    pop.evolve();
    pop.evaluate_population();

    vector<string> pop_str = {};
    for(int i=0;i<population_size;i++){
        pop_str.push_back(pop.pop[i]->print_active_nodes(false));
    }

    srand(seed);
    manual_seed(seed);
    torch::cuda::manual_seed_all(seed);
    pop2.evolve();
    pop2.evaluate_population();

    for(int i=0;i<population_size;i++){
        if(pop_str[i]!=pop2.pop[i]->print_active_nodes(false)){
            cout<<i<<" inequality during evolution"<<endl;
            return false;
        }
    }

    pop.del_pop();
    pop2.del_pop();

    return true;
}

bool cache_equivalence_test_all(){
    bool ret = true;

    ret = ret & cache_equivalence_test(true);
    if(!ret){
        cout<<"False GP"<<endl;
    }
    ret = ret & cache_equivalence_test(false);
    if(!ret){
        cout<<"False CGP"<<endl;
    }
    return ret;
}



bool arma_normalisation_test(){
    arma::mat tensorX = {{1,2},{3,4},{5,6}};

    arma::vec tensory2 = {{7},{8},{9}};
    arma::mat tensory = tensory2;

    Population_arma pop = Population_arma(10, 1, 1000, -1, -1,
                                          13, 13, true, 1,
                                          3, 3, 1,
                                          2, 32, false, 0.5,
                                          1, 1,
                                          {-1., 1.}, 16,
                                          5,
                                          1, 1, 1, -1, 99999., vector<int>({0,1,2,3}), 100, 42, 0.0, true, false);

    pop.initialise_data(tensorX, tensory);



    bool ret = true;
    arma::mat X = {{-1,-1},{0,0},{1,1}};
    arma::mat tmp = arma::sum(arma::abs(X-pop.X));
    ret = ret && tmp.at(0)<0.001;

    ret = ret && abs(pop.norm[0]-8)<0.001;
    ret = ret && abs(pop.norm[1]-1)<0.001;

    arma::mat mean_normalise = {{3, 4}};
    tmp = arma::sum(arma::abs(mean_normalise - pop.mean_normalise));
    ret = ret && tmp.at(0)<0.001;


    arma::mat std_normalise = {{2, 2}};
    tmp = arma::sum(arma::abs(std_normalise - pop.std_normalise));
    ret = ret && tmp.at(0)<0.001;


    return ret;
}

void run_tests() {
    std::vector<std::string> test_state = {"FAILED", "PASSED"};

    std::string passfail;



    passfail = test_state[thread_reproducibility_arma_test()];
    std::cout<<"Thread reproducibility arma test: "<<passfail<<std::endl;

    passfail = test_state[cache_equivalence_test_all()];
    std::cout<<"Cache equivalence test: "<<passfail<<std::endl;

    passfail = test_state[arma_normalisation_test()];
    std::cout<<"Arma normalisation test: "<<passfail<<std::endl;
}



#endif //DCGP_TESTS_H
