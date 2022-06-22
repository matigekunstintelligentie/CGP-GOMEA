//
// Created by joe on 16-03-21.
//

#include "mutualInformation_arma.h"
#include <unordered_set>
#include "../Individual/cgp_arma.h"
#include <map>

MutualInformation_arma::MutualInformation_arma(bool tree, std::vector<int> ops) {
    MutualInformation_arma::tree = tree;

    MutualInformation_arma::convert_op = {};
    for(int i = 0; i<ops.size(); i++){
        MutualInformation_arma::convert_op.insert(std::pair<int,int>{ops[i], i});
    }
}

vector <int> MutualInformation_arma::unified_number_to_node_int(int gene, int rows, int columns, Individual_arma *ind, int max_ops, int max_inputs){
    if(tree==false){
        return {fast_number_to_node_int(gene, rows, ind)};
    }
    else{
        return fast_number_to_node_int_gp(gene, columns, ind, max_ops, max_inputs);
    }
}

// Maps encoding position to specific position or operator of a node
int MutualInformation_arma::fast_number_to_node_int(int gene, int rows, Individual_arma* ind){
    if(gene<ind->get_rows()*ind->get_columns()){
        int r = int(gene%ind->get_rows());
        int c = int(floor(gene/ind->get_rows()));
        return convert_op[ind->get_op(r,c)];
    }
    else{
        gene -= ind->get_rows() * ind->get_columns();

        if(gene<ind->get_rows() * ind->get_columns() * 2){
            int child_number = gene%2;
            gene = int(floor(gene/2));
            int r = int(gene%ind->get_rows());
            int c = int(floor(gene/ind->get_rows()));
            vector<int> pos = ind->get_node_matrix(r,c).child_positions[child_number];
            return pos[1]*rows + pos[0];
        }
        else{
            gene -= ind->get_rows() * ind->get_columns() * 2;
            vector<int> pos = ind->get_output_node(gene).child_positions[0];
            return pos[1]*rows + pos[0];
        }
    }
}

vector<int> MutualInformation_arma::fast_number_to_node_int_gp(int gene, int columns, Individual_arma *ind, int max_ops, int max_inputs) {
    int current = pow(2,columns - 1);
    int current_column = 0;

    while(true){
        if(gene<current){
            int column = current_column;
            int row = gene;
            if(ind->get_node_matrix(row, column).is_constant){
                return {max_ops + max_inputs, row, column};
            }
            else if(ind->get_node_matrix(row, column).is_input){

                return {ind->get_node_matrix(row, column).input_number + max_ops, row, column};
            }

            return {convert_op[ind->get_op(row, column)], row, column};
        }
        else{
            gene -= current;
            current = int(current * 0.5);
            current_column += 1;
        }
    }
}

void MutualInformation_arma::fast_calc_mutual_information(int subset_size, vector<Individual_arma*> pop, vector<vector<float>> *mi, bool MI_distribution_adjustments, vector<vector<float>> *MI_adjustments, int max_ops, int max_inputs, int max_constants, bool normalise_MI){
    // Only GP mode needs constants. CGP tracks connections only!
    bool use_binning = false;
    unordered_set<float> constants;
    vector<float> constants_v;
    if(tree && max_constants>0){
        use_binning = true;
        constants.reserve(max_constants);
        constants_v.reserve(max_constants);
    }

    int encode_number = 0;
    unordered_map<int, int> values_to_int_map;
    values_to_int_map.reserve(1000000);

    vector<vector<int>> pop_nodes(pop.size(), vector<int>(subset_size, 0.0));

    int rows = pop[0]->get_rows();
    int columns = pop[0]->get_columns();

    for(int p = 0;p<pop.size();p++){

        vector<int> nodes;

        for(int i=0;i<subset_size;i++){
            // retrieve node mapping number
            int v;
            vector<int> ret = unified_number_to_node_int(i, rows, columns, pop[p], max_ops, max_inputs);
            v = ret[0];

            // If binning is used and the node mapping number refers to a constant
            if(use_binning && v==max_ops+max_inputs){
                // Check whether there is room left for binning
                if(constants.size()<max_constants){
                    constants.insert(pop[p]->get_node_matrix(ret[1],ret[2]).value);

                    if(constants.size() >= max_constants && constants_v.empty()){

                        //TODO: WARNING!!!! CAN CAUSE DUPLICATES
                        constants_v.insert(constants_v.begin(), constants.begin(), constants.end());
                        sort(constants_v.begin(), constants_v.end());
                    }
                }
                else{
                    const float q = pop[p]->get_node_matrix(ret[1],ret[2]).value;
                    float least_dist = constants_v[0];
                    int least_idx = 0;

                    for(int const_idx = 1; const_idx<constants_v.size(); const_idx++){
                        float dist = abs(q-constants_v[const_idx]);
                        if(dist<least_dist){
                            least_dist = dist;
                            least_idx = const_idx;
                        }
                        else{
                            break;
                        }
                    }

                    v += constants_v[least_idx];
                };
            }


            auto it = values_to_int_map.find(v);
            if (it == values_to_int_map.end()) {
                values_to_int_map[v] = encode_number;

                nodes.push_back(encode_number);

                encode_number++;
            }
            else{
                nodes.emplace_back(it->second);
            }
        }
        pop_nodes[p] = nodes;
    }

    vector<vector<float>> frequencies(values_to_int_map.size(), vector<float>(values_to_int_map.size(), 0.f));
    int val_i, val_j;

    for(int i = 0; i<subset_size;i++){
        for(int j = i+1; j<subset_size;j++){
            for(int p = 0; p<pop.size(); p++){
                val_i = pop_nodes[p][i];
                val_j = pop_nodes[p][j];

                frequencies.at(val_i).at(val_j) += 1.0;
            }

            float freq;
            for(int k=0; k<encode_number;k++){
                for(int l=0; l<encode_number;l++){
                    freq = frequencies.at(k).at(l);
                    if(freq>0.0){
                        freq = freq/pop.size();
                        mi->at(i).at(j) += -freq * log(freq);
                        frequencies.at(k).at(l) = 0.0;
                    }
                }
            }
            mi->at(j).at(i) = mi->at(i).at(j);
        }

        for(int p = 0; p<pop.size(); p++) {
            val_i = pop_nodes[p][i];
            frequencies.at(val_i).at(val_i) += 1.0;
        }

        float freq;
        for(int k=0; k<encode_number;k++) {
            for (int l = 0; l < encode_number; l++) {
                freq = frequencies.at(k).at(l);
                if(freq>0.0){
                    freq = freq/pop.size();
                    mi->at(i).at(i) += -freq * log(freq);
                    frequencies.at(k).at(l) = 0.0;
                }
            }
        }
    }

    if(!MI_distribution_adjustments){
        for(int i = 0; i<subset_size;i++) {
            for (int j = i + 1; j < subset_size; j++) {
                float joint_entropy =  mi->at(i)[j];
                mi->at(i)[j] = mi->at(i)[i] + mi->at(j)[j] - mi->at(i)[j];
                if(normalise_MI){
                    if(joint_entropy==0.0){
                        mi->at(i)[j] = 0.0;
                    }
                    else{
                        mi->at(i)[j] = mi->at(i)[j]/joint_entropy;
                    }
                }

                mi->at(j)[i] = mi->at(i)[j];
            }
        }
    }
    else{
        if(MI_adjustments->size()==0){
            *MI_adjustments = vector<vector<float>>(subset_size, vector<float>(subset_size, 0.0));
            for(int i = 0; i<subset_size;i++){
                MI_adjustments->at(i).at(i) = 1.0/mi->at(i).at(i);
                for(int j = i+1; j<subset_size;j++){
                    MI_adjustments->at(i).at(j) = 2.0/mi->at(i).at(j);
                }
            }
        }

        for(int i = 0; i<subset_size;i++){
            mi->at(i).at(i) = mi->at(i).at(i) * MI_adjustments->at(i).at(i);
            for(int j = i+1; j<subset_size;j++){
                mi->at(i).at(j) = mi->at(i).at(j) * MI_adjustments->at(i).at(j);
            }
        }

        for(int i = 0; i<subset_size;i++){
            for(int j = i+1; j<subset_size;j++){
                float joint_entropy =  mi->at(i)[j];
                mi->at(i).at(j) = mi->at(i).at(i) + mi->at(j).at(j) - mi->at(i).at(j);
                if(normalise_MI) {
                    if(joint_entropy==0.0){
                        mi->at(i)[j] = 0.0;
                    }
                    else{
                        mi->at(i)[j] = mi->at(i)[j]/joint_entropy;
                    }
                }
                mi->at(j).at(i) = mi->at(i).at(j);
            }
        }

    }

}





