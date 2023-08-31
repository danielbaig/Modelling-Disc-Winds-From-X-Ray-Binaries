#ifndef LINEMINIMISER_H
#define LINEMINIMISER_H

#include "extraMaths.h"

template <typename T, typename U>
std::vector<std::vector<T>> operator/(const std::vector<std::vector<T>>& vec2d, U& denom);

template <class T, class Q>
vector <T> operator*(const Q constant, const vector<T> &vector);

template <typename T, typename U>
std::vector<T> operator/(const std::vector<T>& vec, U& denom);

// https://www.javatpoint.com/adding-vectors-in-cpp
template <typename T>
std::vector<T> operator+(const std::vector<T>& vec1, const std::vector<T>& vec2);

// https://www.javatpoint.com/adding-vectors-in-cpp
template <typename T>
std::vector<T> operator-(const std::vector<T>& vec1, const std::vector<T>& vec2);

double getPotential(const std::vector<double> r_point,
                     const std::vector<double> r_potential,
                     const double dRatio);

#endif
