//
// Created by joe on 07-10-21.
//

#include "gom_arma.h"
#include <random>
using namespace std;

gom_arma::gom_arma(bool fix_output_nodes, int max_constants, bool tree_mode) {
    gom_arma::fix_output_nodes = fix_output_nodes;
    gom_arma::tree_mode = tree_mode;
    gom_arma::max_constants = max_constants;
}

void gom_arma::fast_override_nodes_gp(Individual_arma *offspring, Individual_arma *donor, vector<int> *subset, int columns) {

    for(int i=0; i<subset->size(); i++){
        int gene = subset->at(i);

        int current = pow(2,columns - 1);
        int current_column = 0;

        while(true) {
            if (gene < current) {
                int column = current_column;
                int row = gene;

                Node_arma donor_node = donor->get_node_matrix(row, column);

                offspring->set_op(row, column, donor_node.op);
                offspring->set_arity(row, column, donor_node.arity);
                offspring->set_input_number(row, column, donor_node.input_number);
                offspring->set_type(row, column, donor_node.is_input, donor_node.is_constant, donor_node.is_output);

                if(donor_node.is_constant){
                    offspring->set_node_matrix(row, column, donor_node.value, {row, column});

                    offspring->clear_child_positions(row,column);
                    //TODO: why again?
                    for(int child_pos = 0; child_pos<donor_node.child_positions.size();child_pos++){
                        offspring->set_child_positions(row,column,donor_node.child_positions[child_pos]);
                    }
                }
                break;
            }
            else{
                gene -= current;
                current = int(current * 0.5);
                current_column += 1;
            }
        }
    }
}


// Overrides nodes in individual with donor
void gom_arma::fast_override_nodes(Individual_arma* offspring, Individual_arma* donor, vector<int> *subset, int rows, int columns, int n_outputs) {
    for(int i=0; i<subset->size(); i++){
        int gene = subset->at(i);

        // Operator gene
        int r;
        int c;
        if(gene < rows * columns){
            r = int(gene%rows);
            c = int(floor(float(gene)/rows));
            offspring->set_op(r,c, donor->get_node_matrix(r,c).op);
            offspring->set_arity(r,c, donor->get_node_matrix(r,c).arity);
        }
        else{
            gene -= rows * columns;

            // Connection gene
            if(gene<rows*columns*2){
                int child_number = gene%2;
                gene = int(floor(float(gene)/2.));
                r = int(gene%rows);
                c = int(floor(float(gene)/rows));

                vector<int> pos = donor->get_node_matrix_child_pos(r,c,child_number);
                offspring->set_node_matrix_child_pos(r,c,child_number,pos);
            }
            else if(gene<(rows*columns*2 + n_outputs)){
                gene -= rows*columns*2;
                //TODO output too!
                vector<int> pos = donor->get_output_child_pos(gene);


                offspring->set_output_child_pos(gene, pos);

            }
            else{
                gene -= (rows*columns*2 + n_outputs);
                vector<int> pos = donor->get_constant_pos(gene);
                offspring->set_ERC(pos, donor->get_ERC(pos));
            }
        }
    }
}

Individual_arma* gom_arma::fast_mix(int ind, vector<vector<int>> fos, int seed, arma::mat *X, arma::mat *y, vector<Individual_arma* > pop, int *rows, int *columns, int *n_outputs, int *fos_truncation) {
    // Create backup and offspring clone of original individual
    Individual_arma* backup = pop[ind]->clone();
    Individual_arma* offspring = pop[ind]->clone();

    // Create vector of indices of fos
    vector<int> permut_fos_indices;
    permut_fos_indices.reserve(fos.size());
    for (size_t i = 0; i < fos.size(); i++) {
        permut_fos_indices.push_back(i);
    }

    // Shuffle fos indices
    std::mt19937 generator(seed);
    shuffle(permut_fos_indices.begin(), permut_fos_indices.end(), generator);
    uniform_int_distribution<int> distribution(0,pop.size()-1);

    // Truncate FOS and ensure that truncation level lower than fos size
    int truncation = fos.size();
    if(*fos_truncation>-1){
        truncation = min(*fos_truncation, truncation);
    }

    for(int i = 0;i<truncation;i++){
        Individual_arma* donor = pop[distribution(generator)]->clone();
        string active_nodes = offspring->print_active_nodes(false);

        if (tree_mode) {
            fast_override_nodes_gp(offspring, donor, &fos.at(permut_fos_indices[i]), *columns);
        } else {
            fast_override_nodes(offspring, donor, &fos.at(permut_fos_indices[i]), *rows, *columns, *n_outputs);
        }

        string active_nodes_after = offspring->print_active_nodes(false);
        if (active_nodes_after != active_nodes) {
            offspring->evaluate(*X, *y);
        }

        delete donor;

        if(offspring->get_fitness() <= backup->get_fitness()){
            delete backup;
            backup = offspring->clone();
        }
        else{
            delete offspring;
            offspring = backup->clone();

        }
    }

    delete backup;
    return offspring;
}
