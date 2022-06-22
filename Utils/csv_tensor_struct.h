//
// Created by joe on 14-04-21.
//

#ifndef DCGP_CSV_TENSOR_STRUCT_H
#define DCGP_CSV_TENSOR_STRUCT_H

struct csv_tensor{
    csv_tensor(int rows, int cols, std::vector<float> X, std::vector<float> y){
        this->rows = rows;
        this->columns = cols;
        this->X = X;
        this->y = y;
    }
    int rows;
    int columns;
    std::vector<float> X;
    std::vector<float> y;
};

#endif //DCGP_CSV_TENSOR_STRUCT_H
