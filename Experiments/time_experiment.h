//
// Created by joe on 11/3/21.
//

#ifndef DCGP_TIME_EXPERIMENT_H
#define DCGP_TIME_EXPERIMENT_H

using namespace std;

struct experiment_arma_time{
    experiment_arma_time(bool use_tree, bool use_gom, bool apply_MI_adjustments, int fos_type, int generations,
                           int rows, int columns, float percentage_grow, int levels_back, int truncate_fos, std::vector<int> use_indices, int population_size, bool use_erc , bool mix_ERCs , string experiment_name){
        this->use_tree = use_tree;
        this->use_gom = use_gom;
        this->apply_MI_adjustments = apply_MI_adjustments;
        this->fos_type = fos_type;
        this->generations = generations;

        this->rows = rows;
        this->columns = columns;
        this->percentage_grow = percentage_grow;
        this->levels_back = levels_back;
        this->truncate_fos = truncate_fos;
        this->use_indices = use_indices;
        this->population_size = population_size;
        this->experiment_name = experiment_name;
        this->use_erc = use_erc;
        this->mix_ERCs = mix_ERCs;
    }

    bool use_tree;
    bool use_gom;
    bool apply_MI_adjustments;

    int fos_type;
    int generations;
    int rows;
    int columns;

    float percentage_grow;

    int levels_back;
    int truncate_fos;

    std::vector<int> use_indices;

    int population_size;
    string experiment_name;

    bool use_erc;
    bool mix_ERCs;

    string to_string() {
        string use_ind_str = "";
        for(int i=0; i<use_indices.size();i++){
            use_ind_str += std::to_string(use_indices[i]);
            if(i!=use_indices.size()){
                use_ind_str += ",";
            }
        }

        return std::to_string(int(population_size)) + "\t" + std::to_string(int(use_tree)) + "\t" +
        std::to_string(use_gom) + "\t" + std::to_string(apply_MI_adjustments) + "\t" +
        std::to_string(fos_type) + "\t" + std::to_string(rows) + "\t" + std::to_string(columns) + "\t" +
        std::to_string(levels_back) + "\t" + std::to_string(percentage_grow) + "\t" +
        std::to_string(truncate_fos) + "\t" + std::to_string(mix_ERCs) + "\t" + use_ind_str;
    }
};


void run_classic_experiments_arma_time() {
    int repetitions = 30;
    //pair<int, string>(13, "../data/datasets/boston.csv"),
    vector<pair<int, string>> datasets = {pair<int, string>(13, "./data/datasets/boston.csv"), pair<int, string>(6, "./data/datasets/yacht.csv"), pair<int, string>(25, "./data/datasets/tower.csv")};
    std::vector<experiment_arma_time> experiments = {
            experiment_arma_time(true, true, true, 1, 2000000, 8, 4, 0.5, 4, -1, {0,1,2,5,6,7,8,9,10,13,17}, 500, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(true, true, true, 1, 2000000, 8, 4, 0.5, 4, -1, {0,1,2,5,6,7,8,9,10,13,17}, 1000, false, false,"./data/experiments/popsizing.csv"),
            experiment_arma_time(true, true, true, 1, 2000000, 8, 4, 0.5, 4, -1, {0,1,2,5,6,7,8,9,10,13,17}, 2000, false, false,"./data/experiments/popsizing.csv"),
            experiment_arma_time(true, true, true, 1, 2000000, 8, 4, 0.5, 4, -1, {0,1,2,5,6,7,8,9,10,13,17}, 4000, false, false,"./data/experiments/popsizing.csv"),
            experiment_arma_time(true, true, true, 1, 2000000, 8, 4, 0.5, 4, -1, {0,1,2,5,6,7,8,9,10,13,17}, 8000, false, false,"./data/experiments/popsizing.csv"),
            experiment_arma_time(true, true, true, 1, 2000000, 8, 4, 0.5, 4, -1, {0,1,2,5,6,7,8,9,10,13,17}, 16000, false, false,"./data/experiments/popsizing.csv"),
            experiment_arma_time(true, true, true, 1, 2000000, 8, 4, 0.5, 4, -1, {0,1,2,5,6,7,8,9,10,13,17}, 32000, false, false,"./data/experiments/popsizing.csv"),
            experiment_arma_time(true, true, true, 1, 2000000, 8, 4, 0.5, 4, -1, {0,1,2,5,6,7,8,9,10,13,17}, 64000, false, false,"./data/experiments/popsizing.csv"),
            experiment_arma_time(true, true, true, 1, 2000000, 8, 4, 0.5, 4, -1, {0,1,2,5,6,7,8,9,10,13,17}, 128000, false, false,"./data/experiments/popsizing.csv"),
            experiment_arma_time(true, true, true, 1, 2000000, 8, 4, 0.5, 4, -1, {0,1,2,5,6,7,8,9,10,13,17}, 256000, false, false,"./data/experiments/popsizing.csv"),
            experiment_arma_time(true, true, true, 1, 2000000, 8, 4, 0.5, 4, -1, {0,1,2,5,6,7,8,9,10,13,17}, 512000, false, false,"./data/experiments/popsizing.csv"),

            experiment_arma_time(false, true, true, 1, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 500, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, true, true, 1, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 1000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, true, true, 1, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 2000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, true, true, 1, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 4000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, true, true, 1, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 8000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, true, true, 1, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 16000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, true, true, 1, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 32000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, true, true, 1, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 64000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, true, true, 1, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 128000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, true, true, 1, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 256000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, true, true, 1, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 512000, false, false, "./data/experiments/popsizing.csv"),

            experiment_arma_time(false, false, false, 0, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 500, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, false, false, 0, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 1000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, false, false, 0, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 2000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, false, false, 0, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 4000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, false, false, 0, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 8000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, false, false, 0, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 16000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, false, false, 0, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 32000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, false, false, 0, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 64000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, false, false, 0, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 128000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, false, false, 0, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 256000, false, false, "./data/experiments/popsizing.csv"),
            experiment_arma_time(false, false, false, 0, 2000000, 8, 3, 1.0, 3, -1, {0,1,2,5,6,7,8,9,10,13,17}, 512000, false, false, "./data/experiments/popsizing.csv"),

            experiment_arma_time(true, true, true, 1, 2000000, 16, 5, 0.5, 5, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, false, false, "./data/experiments/noerc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 16, 4, 1.0, 4, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, false, false, "./data/experiments/noerc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 16, 4, 1.0, 4, 61, {0,1,2,8,9,10,13,17,7,5,6}, 1000, false, false, "./data/experiments/noerc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 8, 8, 1.0, 8, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, false, false, "./data/experiments/noerc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 1, 10, 1.0, 10, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, false, false, "./data/experiments/noerc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 16, 4, 1.0, 1, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, false, false, "./data/experiments/noerc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 1, 64, 1.0, 1, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, false, false, "./data/experiments/noerc.csv"),
            experiment_arma_time(false, false,false, 0, 2000000, 16, 4, 1.0, 4, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, false, false, "./data/experiments/noerc.csv"),

            experiment_arma_time(true, true, true, 1, 2000000, 16, 5, 0.5, 5, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, true, false, "../data/experiments/erc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 16, 4, 1.0, 4, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, true, true,"./data/experiments/erc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 16, 4, 1.0, 4, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, true, false,"./data/experiments/erc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 16, 4, 1.0, 4, 61, {0,1,2,8,9,10,13,17,7,5,6}, 1000, true,true, "./data/experiments/erc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 8, 8, 1.0, 8, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, true, true,"./data/experiments/erc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 1, 10, 1.0, 10, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, true,true, "./data/experiments/erc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 16, 4, 1.0, 1, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, true, true,"./data/experiments/erc.csv"),
            experiment_arma_time(false, true,true, 1, 2000000, 1, 64, 1.0, 1, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, true, true,"./data/experiments/erc.csv"),
            experiment_arma_time(false, false,false, 0, 2000000, 16, 4, 1.0, 4, -1, {0,1,2,8,9,10,13,17,7,5,6}, 1000, true, true,"./data/experiments/erc.csv")

      };


    std::chrono::time_point<std::chrono::system_clock> t = chrono::system_clock::now();


    int experiment_nr = 0;
    for(auto exp:experiments){
        for(auto dataset:datasets) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < repetitions; i++) {
                int seed = i + 1;
                std::srand(seed);
                torch::manual_seed(seed);
                torch::cuda::manual_seed_all(seed);

                std::vector<arma::mat> splits = load_tensors_arma(dataset.second, 0.75, 0.0);
                std::vector<float> max_value = get_ERC_vals_arma(splits[0]);

                float time_limit = 5000;

                cout<<"Experiment "<<experiment_nr<<"/"<<repetitions*experiments.size()*datasets.size()<<endl;
                float time_spent = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - t).count()/(3600.*1000.);
                cout<<"Time "<<(experiment_nr + 1)*time_limit/3600.<<"/"<<repetitions*experiments.size()*datasets.size()*time_limit/3600.<<" time spent: "<<time_spent<<endl;



                Population_arma pop = Population_arma(exp.population_size, exp.generations, time_limit, -1, -1,
                                                      dataset.first, 31, exp.use_erc, 1,
                                                      exp.rows, exp.columns, exp.levels_back,
                                                      2, 16, exp.use_tree, exp.percentage_grow,
                                                      true, false,
                                                      max_value, 4,
                                                      4,
                                                      exp.use_gom,
                                                      exp.fos_type, exp.apply_MI_adjustments, exp.truncate_fos, 999999,
                                                      exp.use_indices, 100, seed, 0., true, false);


                pop.initialise_data(splits[0], splits[1]);
                pop.initialise_population();

                pop.evolve();

                #pragma omp critical
                {


                    final_evaluation_classic(exp.experiment_name, pop, splits, std::to_string(experiment_nr + i) + "\t" + exp.to_string() + "\t" + dataset.second,
                                             true, false);
                }

                pop.del_pop();
            }
        }
    }
}

#endif //DCGP_TIME_EXPERIMENT_H
