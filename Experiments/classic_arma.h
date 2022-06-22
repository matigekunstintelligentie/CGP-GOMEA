//
// Created by joe on 08-10-21.
//

#ifndef DCGP_CLASSIC_ARMA_H
#define DCGP_CLASSIC_ARMA_H
using namespace std;

static void final_evaluation_classic(string filename, Population_arma pop, vector<arma::mat> splits, string const experiment, bool csv, bool linear_after){
    pop.test_mode(true);
    pop.sort_population();

    cout<<"Top individual on training set:"<<endl;
    cout<<pop.pop[0]->print_active_nodes(false)<<endl;

    // Normalise inputs using mean and std found for training split
    if(pop.use_normalise){
        splits[0] = (splits[0] - pop.mean_normalise)/pop.std_normalise;
        splits[2] = (splits[2] - pop.mean_normalise)/pop.std_normalise;
        splits[4] = (splits[4] - pop.mean_normalise)/pop.std_normalise;
        splits[6] = (splits[6] - pop.mean_normalise)/pop.std_normalise;
    }

    // Get normalisation and scaling constants from training
    vector<float> norm = pop.norm;

    vector<float> a_b = pop.pop[0]->get_ab(splits[0], splits[1]);

    cout<<"Normalisation variables: "<< norm[0]<<", "<<norm[1]<< endl;
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

    res+= "\t" + to_string(test_var);

    cout<<"R2: ";
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

        res += "\t" + to_string(validation_var);
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

    res+= "\t" + to_string(training_var);

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

    res+= "\t" + to_string(total_var);

    cout << "R2: ";
    cout<<1.-pop.pop[0]->get_fitness()/total_var<<endl;
    res+= "\t" + to_string(1.-pop.pop[0]->get_fitness()/total_var);

    cout<<"Number of nodes total: ";
    cout<<pop.pop[0]->get_node_number(false)<<endl;
    res+= "\t" + to_string(pop.pop[0]->get_node_number(false)[0][0]) + "\t" + to_string(pop.pop[0]->get_node_number(false)[0][1]) + "\t" + to_string(pop.pop[0]->get_node_number(false)[0][2]);

    cout<<"Number of nodes unique: ";
    cout<<pop.pop[0]->get_node_number(true)<<endl;
    res+= "\t" + to_string(pop.pop[0]->get_node_number(true)[0][0]) + "\t" + to_string(pop.pop[0]->get_node_number(true)[0][1]) + "\t" + to_string(pop.pop[0]->get_node_number(true)[0][2]);

    res+= "\t" + to_string(pop.pop[0]->get_active_nodes().size());

    res+= "\t" + pop.pop[0]->print_active_nodes(false);

    cout<<"Number of semantic reuse: "<< to_string(pop.pop[0]->get_semantic_duplicates(splits[0]))<<endl;
    res+= "\t" + to_string(pop.pop[0]->get_semantic_duplicates(splits[0]));

    res+= "\t" + to_string(pop.generation);

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

    res += string("\tcpu time\t");

    for(auto cpu_time:pop.times){
        res += to_string(cpu_time);
        res +=  ",";
    }
    res = res.substr(0, res.size()-1);

    if(csv){
        if(filename==""){
            write_csv("../data/cluster_experiment.csv", experiment + "\t" + res + "\n");
        }
        else{
            write_csv(filename, experiment + "\t" + res + "\n");
        }
    }
}

struct experiment_arma{
    experiment_arma(bool use_tree, bool use_gom, bool apply_MI_adjustments, int fos_type, int generations,
               int rows, int columns){
        this->use_tree = use_tree;
        this->use_gom = use_gom;
        this->apply_MI_adjustments = apply_MI_adjustments;
        this->fos_type = fos_type;
        this->generations = generations;

        this->rows = rows;
        this->columns = columns;
    }

    bool use_tree;
    bool use_gom;
    bool apply_MI_adjustments;

    int fos_type;
    int generations;
    int rows;
    int columns;

    string to_string() {
        return std::to_string(int(use_tree)) + "\t" + std::to_string(use_gom) + "\t" + std::to_string(apply_MI_adjustments) + "\t" + std::to_string(fos_type) + "\t" + std::to_string(rows) + "\t" + std::to_string(columns);
    }
};

// Mimics settings from paper table 3
// 10 Datasets
// Median of 30 runs
// 20 Generations
// Population of 2000
// No Constants
// CGP, CGP classic, CGP GPmode, GP
void run_classic_experiments_arma() {
    vector<int> population_sizes = {128,256,512,1024,2048,4096};
    int repetitions = 30;
    vector<pair<int, string>> datasets = {pair<int, string>(13, "../data/datasets/boston.csv"),
            pair<int, string>(6, "../data/datasets/yacht.csv"),
                                          pair<int, string>(5, "../data/datasets/air.csv"),
                                          pair<int, string>(8, "../data/datasets/concrete.csv"),
                                          pair<int, string>(57, "../data/datasets/dowchemical.csv"),
                                          pair<int, string>(8, "../data/datasets/energycooling.csv"),
                                          pair<int, string>(8, "../data/datasets/energyheating.csv"),
                                          pair<int, string>(11, "../data/datasets/winequality-red.csv"),
                                          pair<int, string>(11, "../data/datasets/winequality-white.csv"),
                                          pair<int, string>(25, "../data/datasets/tower.csv")};
    std::vector<experiment_arma> experiments = {
            experiment_arma(true, true, true, 1, 20, 16, 5),
            experiment_arma(true, true, false, 1, 20, 16, 5),
            experiment_arma(true, true, false, 0, 20, 16, 5),

            experiment_arma(false, true,true, 1, 20, 16, 4),
            experiment_arma(false, true,false, 1, 20, 16, 4),
            experiment_arma(false, true,false, 0, 20, 16, 4),

            experiment_arma(false, true,true, 1, 20, 16, 4),
            experiment_arma(false, true,false, 1, 20, 16, 4),
            experiment_arma(false, true,false, 0, 20, 16, 4),
            experiment_arma(false, true,true, 1, 20, 8, 8),

            experiment_arma(false, true,true, 1, 20, 1, 16),

            experiment_arma(false, false,false, 0, 1200, 16, 5)
    };

    for(auto exp:experiments){
        for(auto dataset:datasets) {
            for (int i = 0; i < repetitions; i++) {
                std::srand(i + 1);
                torch::manual_seed(i + 1);
                torch::cuda::manual_seed_all(i + 1);

                std::vector<arma::mat> splits = load_tensors_arma(dataset.second, 0.5, 0.25);
                std::vector<float> max_value = get_ERC_vals_arma(splits[0]);

                Population_arma pop = Population_arma(2000, exp.generations, -1, -1, -1,
                                            dataset.first, 0, false, 1,
                                            exp.rows, exp.columns, exp.columns,
                                            2, 32, exp.use_tree, 0.5,
                                            true, false,
                                            max_value, 4,
                                            4,
                                            exp.use_gom,
                                            exp.fos_type, exp.apply_MI_adjustments, -1, 999999,
                                            {0,1,2,3}, 100, i+1, 0.0, true, false);

                pop.initialise_data(splits[0], splits[1]);
                pop.initialise_population();

                pop.evolve();

                string experiment_name = "../data/experiments/classic_arma.csv";

                final_evaluation_classic(experiment_name, pop, splits, exp.to_string() + "\t" + dataset.second, true, false);

                pop.del_pop();
            }
        }
    }
}
#endif //DCGP_CLASSIC_ARMA_H
