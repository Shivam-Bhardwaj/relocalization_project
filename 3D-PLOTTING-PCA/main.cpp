#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <bits/stdc++.h>
#include "Eigen/Dense"
#include <string>

using namespace std;
using namespace Eigen;

Eigen::Vector4d NormalizeQuaternion(const Eigen::Vector4d &qvec) {
    return qvec;
}

Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Vector4d &qvec) {
    const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(qvec);
    const Eigen::Quaterniond quat(normalized_qvec(0), normalized_qvec(1),
                                  normalized_qvec(2), normalized_qvec(3));
    return quat.toRotationMatrix();
}


int main() {
    string line;
    int count = 0;
    vector<string> single;
    ofstream fileo;
    fileo.open("../7DOF_to_3D.txt");
    ofstream fileo_rot;
    fileo_rot.open("../rot.txt");
    ifstream ifile;
    ifile.open("../images.txt");
    cout << "Reading data from a file :-" << endl << endl;

    while (ifile) {
        getline(ifile, line);
        if (line[0] != '#') {
            count += 1;
            if ((count % 2)) {
                boost::split(single, line, [](char c) { return c == ' '; });
                Vector4d qvec(stof(single[1]), stof(single[2]), stof(single[3]), stof(single[4]));
                Vector3d trans(stof(single[5]), stof(single[6]), stof(single[7]));
                Matrix3d rot = QuaternionToRotationMatrix(qvec);
                Vector3d unit(1.0,1.0,1.0);
                fileo << ((-1 * rot).transpose()) * trans << endl;
                fileo_rot << rot*unit<<endl;
            }

        }
    }
    cout << "Reading complete" << endl;
    ifile.close();
    fileo.close();
    fileo_rot.close();

    return 0;
}
