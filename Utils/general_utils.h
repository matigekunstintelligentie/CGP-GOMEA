//
// Created by joe on 6/8/21.
//

#ifndef DCGP_GENERAL_UTILS_H
#define DCGP_GENERAL_UTILS_H

float findMedian(std::vector<float> a, int n)
{
    if(n==0){
        return 0.0;
    }
    // First we sort the array
    std::sort(a.begin(), a.end());

    // check for even case
    if (n % 2 != 0) {
        return (float) a[n / 2];
    }

    return (float)(a[(n - 1) / 2] + a[n / 2]) / 2.0;
}



std::vector<float> get_ERC_vals(torch::Tensor tensorX){
    torch::Tensor min = torch::min(tensorX);
    float max = torch::max(tensorX).item<float>();
    return {min.item<float>(), max};
}

std::vector<float> get_ERC_vals_arma(arma::mat tensorX){
    arma::mat min = arma::min(tensorX);
    arma::mat max = arma::max(tensorX);
    return {(float) min.at(0), (float) max.at(0)};
}

#endif //DCGP_GENERAL_UTILS_H
