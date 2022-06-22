//
// Created by joe on 01-04-21.
//

#include "fos_arma.h"
#include "upgma.h"
#include "mutualInformation_arma.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>

Fos_arma::Fos_arma(bool apply_MI_adjustments, int max_constants, bool tree_mode){
    Fos_arma::apply_MI_adjustments = apply_MI_adjustments;
    Fos_arma::max_constants = max_constants;
    Fos_arma::tree_mode = tree_mode;
}

void Fos_arma::build_mutual_information_fos(std::vector<std::vector<float>> *muinf, int univariate_size) {

    std::vector<std::vector<int>> r_fos = calc_upgma_marco(*muinf, univariate_size);

    subsets = r_fos;
}

void Fos_arma::build_random_marco(int univariate_size) {
    //TODO: dont forget mechanic!
    std::vector<std::vector<int>> FOS;
    for (int i = 0; i < univariate_size; i++) {
        std::vector<int> subset = {i};
        FOS.push_back(subset);
    }
    std::vector<std::vector<int>> fos_clone(FOS);
    while (fos_clone.size() > 2) {
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        int first_set_index = r * fos_clone.size();
        int second_set_index;
        do {
            r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            second_set_index = r * fos_clone.size();
        } while (second_set_index == first_set_index);

        if (first_set_index > second_set_index) {
            int temp = first_set_index;
            first_set_index = second_set_index;
            second_set_index = temp;
        }

        std::vector<int> merged_set;
        merged_set.reserve(fos_clone[first_set_index].size() + fos_clone[second_set_index].size());

        for (int v : fos_clone[first_set_index]) {
            merged_set.push_back(v);
        }

        for (int v : fos_clone[second_set_index]) {
            merged_set.push_back(v);
        }

        fos_clone.erase(fos_clone.begin() + first_set_index);
        fos_clone.erase(fos_clone.begin() + second_set_index - 1);

        fos_clone.push_back(merged_set);
        FOS.push_back(merged_set);

    }
    subsets = FOS;
}

void Fos_arma::build_random_fos(int univariate_size) {
    std::vector<std::vector<float>> M = {};
    for(int i=0; i<univariate_size; i++){
        std::vector<float> row;
        for(int j=0; j<univariate_size;j++){
            row.emplace_back(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
        }
        M.emplace_back(row);
    }

    std::vector<std::vector<int>> r_fos = calc_upgma_marco(M,univariate_size);

    subsets = r_fos;
}


void Fos_arma::build_univariate_fos(int univariate_size){
    std::vector<std::vector<int>> r_fos = {};

    for(int i = 0; i<univariate_size; i++){
        r_fos.emplace_back(std::vector<int>({i}));
    }

    subsets = r_fos;
};

void Fos_arma::replace_MI(std::vector<std::vector<float>> M, int size){
    std::vector<std::vector<int>> r_fos = calc_upgma_marco(M,size);

    subsets = r_fos;
}

void process_mem_usage(double& vm_usage, double& resident_set)
{
    vm_usage     = 0.0;
    resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;
}

void Fos_arma::build_fos(int fos_type, int rows, int columns, int n_outputs, std::vector<Individual_arma*> pop, std::vector<std::vector<float>> *MI_adjustments, int max_ops, int max_inputs, bool normalise_MI, std::vector<int> ops, bool mix_ERCs){

    // 3 = (1+max arity)
    int univariate_size = rows*columns*3 + n_outputs;

    //formula: (max_arity**(cols+1) - 1)/(max_arity-1)
    if(tree_mode) {
        univariate_size = (pow(2, columns) - 1) / (1);
    }

//    double vm, rss;
//    process_mem_usage(vm, rss);
//    std::cout << "VM: " << vm << "; RSS: " << rss << std::endl;
//    std::cout<<"FOS size: "<<subsets.size()<<std::endl;

    switch (fos_type) {
        case 0: {
            build_random_fos(univariate_size);
            break;
        }
        case 1: {

            MutualInformation_arma mi = MutualInformation_arma(tree_mode, ops);

            std::vector<std::vector<float>> muinf(univariate_size, std::vector<float>(univariate_size, 0.));

            auto t = std::chrono::system_clock::now();
            mi.fast_calc_mutual_information(univariate_size, pop, &muinf, apply_MI_adjustments, MI_adjustments, max_ops, max_inputs, max_constants, normalise_MI);

            build_mutual_information_fos(&muinf, univariate_size);

            if(mix_ERCs){
                for(int i=0;i<pop[0]->get_n_constants();i++){
                    subsets.push_back(std::vector<int>({univariate_size+i}));
                }
            }

//            std::ofstream f ("MIs.txt", std::ios::app);
//            if(f.is_open()){
//                for(int i =0;i<muinf.size();i++){
//                    for(int j =0;j<muinf[i].size();j++){
//                        if(j==muinf[i].size()-1){
//                            f << muinf[i][j];
//                        }
//                        else{
//                            f << muinf[i][j] << ",";
//                        }
//
//                    }
//                    f << "\n";
//                }
//            }
//            f << "-\n";
//            f.close();


            break;
        }
        case 2: {
            build_univariate_fos(univariate_size);
            break;
        }
        case 3: {
            build_random_marco(univariate_size);
            break;
        }
        default: {
            break;
        }
    };

};
