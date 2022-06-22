//
// Created by joe on 01-12-21.
//

#ifndef DCGP_SMALL_TREE_H
#define DCGP_SMALL_TREE_H

using namespace std;

// Mimics settings from paper table 3
// 10 Datasets
// Median of 30 runs
// 20 Generations
// Population of 2000
// No Constants
// CGP, CGP classic, CGP GPmode, GP
void run_classic_experiments_arma_small_tree() {
    int repetitions = 30;
    vector<pair<int, string>> datasets = {
            pair<int, string>(4, "../data/datasets/fake_data.csv")};


    std::vector<experiment_arma_time> experiments = {
            experiment_arma_time(false, false, false, 0, 2000000, 16, 4, 1.0, 4, -1, {0,1,2,8}, 1000, false, false,"../data/experiments/knownformula.csv"),
            experiment_arma_time(false, true, true, 1, 2000000, 16, 4, 1.0, 4, -1, {0,1,2,8}, 1000, false, false,"../data/experiments/knownformula.csv"),
            experiment_arma_time(true, true, true, 1, 2000000, 16, 5, 0.5, 5, -1, {0,1,2,8}, 1000, false, false,"../data/experiments/knownformula.csv"),
            experiment_arma_time(false, true, true, 1, 2000000, 16, 4, 1.0, 1, -1, {0,1,2,8}, 1000, false,false,"../data/experiments/knownformula.csv")
                };

    int experiment_nr = 0;
    for(auto exp:experiments){
        for(auto dataset:datasets) {
            //#pragma omp parallel for schedule(static)
            for (int i = 0; i < repetitions; i++) {
                std::srand(i + 1);
                torch::manual_seed(i + 1);
                torch::cuda::manual_seed_all(i + 1);

                std::vector<arma::mat> splits = load_tensors_arma(dataset.second, 0.75, 0.0);
                std::vector<float> max_value = get_ERC_vals_arma(splits[0]);

                cout<<"Experiment "<<experiment_nr<<"/"<<repetitions*experiments.size()*datasets.size()<<endl;

                experiment_nr += 1;
                Population_arma pop = Population_arma(exp.population_size, exp.generations, 5000, -1, -1,
                                                      dataset.first, 0, exp.use_erc, 1,
                                                      exp.rows, exp.columns, exp.levels_back,
                                                      2, 32, exp.use_tree, exp.percentage_grow,
                                                      false, false,
                                                      max_value, 4,
                                                      4,
                                                      exp.use_gom,
                                                      exp.fos_type, exp.apply_MI_adjustments, exp.truncate_fos, 999999,
                                                      exp.use_indices, 100, i + 1, 0., true, false);

                pop.eps = 10e-6;
                pop.initialise_data(splits[0], splits[1]);
                pop.initialise_population();

                pop.evolve();


                final_evaluation_classic(exp.experiment_name, pop, splits, exp.to_string() + "\t" + dataset.second, true, false);

                pop.del_pop();
            }
        }
    }
}

#endif //DCGP_SMALL_TREE_H
