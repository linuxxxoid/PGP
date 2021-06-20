#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <iomanip>
#include <string.h>
#include <chrono>

using namespace std;
using namespace std::chrono;


//#define ndims 3
//#define ndims_x_2 6

const int ndims = 3;
const int ndims_x_2 = 6;

double u_next(double ux0, double ux1, double uy0, double uy1, 
                        double uz0, double uz1, double h2x, 
                        double h2y, double h2z){
    double ans = (ux0 + ux1) * h2x;
    ans += (uy0 + uy1) * h2y; 
    ans += (uz0 + uz1) * h2z;
    return ans;
}

double max_determine(double val1, double val2, double curr_max){
    double diff = val1 - val2;
    diff = diff < 0.0 ? -diff : diff;

    return diff > curr_max ? diff : curr_max;
}


void print_line(ostream& os, double* line, int size){
    for(int i = 0; i < size; ++i){
        os << line[i] << " ";
    }
}


int main(int argc, char **argv){
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);


    // input data
    int dimens[ndims], blocks[ndims];
    double l[ndims];
    double u[ndims_x_2];
    double u0, eps;
    string path;

    // num of each orientation and direction
    enum orientation{
        left = 0, right,
        front, back,
        down, up,
    };
    enum direction{
        dir_x = 0,
        dir_y,
        dir_z
    };

    cin >> dimens[dir_x] >> dimens[dir_y] >> dimens[dir_z];
    cin >> blocks[dir_x] >> blocks[dir_y] >> blocks[dir_z];
    cin >> path;
    cin >> eps;
    cin >> l[dir_x] >> l[dir_y] >> l[dir_z];
    cin >> u[down] >> u[up];
    cin >> u[left] >> u[right];
    cin >> u[front] >> u[back];
    cin >> u0;

    auto start = steady_clock::now();

    double max_diff = 0.0;

    int nx = dimens[dir_x]*blocks[dir_x];
    int ny = dimens[dir_y]*blocks[dir_y];
    int nz = dimens[dir_z]*blocks[dir_z];

    int sizex = nx + 2;
    int sizey = ny + 2;
    int sizez = nz + 2;

    double h2x, h2y, h2z; 
    h2x = l[dir_x] / ((double)nx);
    h2y = l[dir_y] / ((double)ny);
    h2z = l[dir_z] / ((double)nz);

    h2x *= h2x;
    h2y *= h2y;
    h2z *= h2z;

    {
        double denuminator = 2.0*(1.0/h2x + 1.0/h2y + 1.0/h2z);
        h2x = 1.0 / (denuminator * h2x);
        h2y = 1.0 / (denuminator * h2y);
        h2z = 1.0 / (denuminator * h2z);
    }

    double* buffer0 = new double[sizex * sizey * sizez]; 
    double* buffer1 = new double[sizex * sizey * sizez];

    fill_n(buffer0, sizex * sizey * sizez, u0);

    int orr = 0;
    for(int i = 0; i < sizex; i += nx + 1, ++orr){
        for(int j = 1; j < ny + 1; ++j){
            for(int k = 1; k < nz + 1; ++k){
                buffer0[i + (j + k*sizey)*sizex] = u[orr];
                buffer1[i + (j + k*sizey)*sizex] = u[orr];
            }
        }
    }

    for(int j = 0; j < sizey; j += ny + 1, ++orr){
        for(int k = 1; k < nz + 1; ++k){
            for(int i = 1; i < nx + 1; ++i){
                buffer0[i + (j + k*sizey)*sizex] = u[orr];
                buffer1[i + (j + k*sizey)*sizex] = u[orr];
            }
        }
    }

    for(int k = 0; k < sizez; k += nz + 1, ++orr){
        for(int j = 1; j < ny + 1; ++j){
            for(int i = 1; i < nx + 1; ++i){
                buffer0[i + (j + k*sizey)*sizex] = u[orr];
                buffer1[i + (j + k*sizey)*sizex] = u[orr];
            }
        }
    }

    do{
        max_diff = 0.0;
        for(int k = 1; k <= nz; ++k){
            for(int j = 1; j <= ny; ++j){
                for(int i = 1; i <= nx; ++i){
                    buffer1[i + (j + k*sizey)*sizex] = u_next(
                        buffer0[i - 1 + (j + k*sizey)*sizex], buffer0[i + 1 + (j + k*sizey)*sizex],
                        buffer0[i + (j - 1 + k*sizey)*sizex], buffer0[i + (j + 1 + k*sizey)*sizex],
                        buffer0[i + (j + k*sizey - sizey)*sizex], buffer0[i + (j + k*sizey + sizey)*sizex],
                        h2x, h2y, h2z
                    );
                    max_diff = max_determine(
                        buffer0[i + (j + k*sizey)*sizex], 
                        buffer1[i + (j + k*sizey)*sizex], 
                        max_diff
                    );
                }
            }
        }

        double* temp = buffer0;
        buffer0 = buffer1;
        buffer1 = temp;
    }while(max_diff >= eps);

    ofstream fout(path, ios::out);
    fout << std::scientific << std::setprecision(7);

    for(int k = 1; k <= nz; ++k){
        for(int j = 1; j <= ny; ++j){
            for(int i = 1; i <= nx; ++i){
                fout << buffer0[i + (j + k*sizey)*sizex] << " ";
            }
        }
    }
    fout << endl;

    delete[] buffer0;
    delete[] buffer1;

    fout.close();

    auto end = steady_clock::now();
    duration<double> elaps = end - start; 
    cout << "TIME: ";
    cout << elaps.count() * 1000.0 << "ms" << endl;

    return 0;
}