//
// Created by joe on 14-04-21.
//

#include "csv_tensor_struct.h"
#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include <armadillo>

#ifndef DCGP_CSV_UTILS_H
#define DCGP_CSV_UTILS_H

static bool is_file_exist(std::string fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

static void write_csv_header(std::string filename, std::string header){
    bool file_exist = is_file_exist(filename);
    std::ofstream csv_file;
    csv_file.open(filename, std::ofstream::out | std::ofstream::app);
    if(!file_exist){
        csv_file << header;
    }
    csv_file.close();
}

static void write_csv(std::string filename, std::string input_string, std::string header){
    bool file_exist = is_file_exist(filename);
    std::ofstream csv_file;
    csv_file.open(filename, std::ofstream::out | std::ofstream::app);
    if(!file_exist){
        csv_file << header;
    }
    csv_file << input_string;
    csv_file.close();
}

static void write_csv(std::string filename, std::string input_string){

    std::ofstream csv_file;
    csv_file.open(filename, std::ofstream::out | std::ofstream::app);
    csv_file << input_string;
    csv_file.close();
}

static csv_tensor load_tensors(std::string filename){
    std::ifstream fin;
    std::string line;
    // Open an existing file
    fin.open(filename);
    int rows;
    int cols;
    while(fin.good()){
        std::getline(fin, line);
        std::stringstream ss(line);
        ss>>rows;
        ss.ignore();
        ss>>cols;
        std::getline(fin, line);
        break;
    }
    //std::cout<<"rows "<<rows<<", columns "<<cols<<std::endl;

    std::vector<float> X;
    std::vector<float> y;

    int curr_row = 0;
    while(std::getline(fin, line)){
        std::stringstream ss(line);
        int curr_col = 0;
        float val;


        while(ss >> val){
            if(curr_col==cols){
                y.emplace_back(val);
                break;
            }
            else{
                X.emplace_back(val);
            }
            if(ss.peek() == ',') ss.ignore();
            curr_col++;
        }
        curr_row++;
    }
    fin.close();

    csv_tensor csv_t(rows, cols, X, y);

    return csv_t;
}

static int get_columns(std::string filename){
    csv_tensor ct = load_tensors(filename);

    return ct.columns;
}



// Reads csv file. First line should have the number of rows and input columns. The last column should be the target value.
// Also slices the data into a training, validation and test set.
static std::vector<torch::Tensor> load_tensors(std::string filename, float train_p, float val_p){
    csv_tensor ct = load_tensors(filename);

//    cout<<ct.X.size()<<" "<<ct.rows<<" "<<ct.columns<<endl;
    at::Tensor tensor_X = torch::from_blob((float*)(ct.X.data()), {ct.rows, ct.columns}, at::kFloat).clone();
    at::Tensor tensor_y = torch::from_blob((float*)(ct.y.data()), {ct.rows, 1}, at::kFloat).clone();

    torch::Tensor perm = torch::randperm(ct.rows, torch::TensorOptions().dtype(torch::kLong));

    torch::Tensor idx_train = torch::slice(perm, 0, 0, int(train_p*ct.rows));

    torch::Tensor tensor_X_train = torch::index(tensor_X, {idx_train});
    torch::Tensor tensor_y_train = torch::index(tensor_y, {idx_train});

    //std::cout<<"Train rows "<<tensor_y_train.size(0)<<std::endl;

    torch::Tensor idx_val = torch::slice(perm, 0, int(train_p*ct.rows), int(train_p*ct.rows)+int(val_p*ct.rows));

    torch::Tensor tensor_X_val = torch::index(tensor_X, {idx_val});
    torch::Tensor tensor_y_val = torch::index(tensor_y, {idx_val});

    //std::cout<<"Val rows "<<tensor_y_val.size(0)<<std::endl;

    torch::Tensor idx_test = torch::slice(perm, 0, int(train_p*ct.rows)+int(val_p*ct.rows), ct.rows);

    torch::Tensor tensor_X_test = torch::index(tensor_X, {idx_test});
    torch::Tensor tensor_y_test = torch::index(tensor_y, {idx_test});

    //std::cout<<"Test rows "<<tensor_y_test.size(0)<<std::endl;

    return {tensor_X_train, tensor_y_train, tensor_X_val, tensor_y_val, tensor_X_test, tensor_y_test, tensor_X, tensor_y};
}

static std::vector<arma::mat> load_tensors_arma(std::string filename, float train_p, float val_p){
    std::vector<torch::Tensor> torch_tensors = load_tensors(filename, train_p, val_p);
    std::vector<arma::mat> arma_tensors = {};

    for(int z = 0; z<torch_tensors.size(); z++){
        arma::mat armay = arma::zeros(torch_tensors[z].size(0),torch_tensors[z].size(1));
        for(int i=0;i<torch_tensors[z].size(0);i++){
            for(int j=0;j<torch_tensors[z].size(1);j++){
                armay(i,j) = torch_tensors[z][i][j].item<float>();
            }
        }
        arma_tensors.push_back(armay);
    }

    return arma_tensors;
}

#endif //DCGP_CSV_UTILS_H
