#include<iostream>
#include<fstream>
#include <boost/algorithm/string.hpp>
#include <bits/stdc++.h>

using namespace std;

int main() {
    string line;
    int count = 0;
    vector<string> single;
    ofstream fileo;
    fileo.open("../new.txt");
    ifstream ifile;
    ifile.open("../images.txt");
    cout << "Reading data from a file :-" << endl << endl;

    while (ifile) {
        getline(ifile, line);
        if (line[0] != '#') {
            count += 1;
            if ((count % 2)) {
                boost::split(single, line, [](char c) { return c == ' '; });
                fileo << single[9] << " " << single[5] << " " << single[6] << " " << single[7] << " "
                      << single[1] << " " << single[2] << " " << single[3] << " " << single[4] << endl;

            }

        }
    }
    cout << "Reading complete" << endl;
    ifile.close();
    fileo.close();

    int train = int(float(count * 0.70 / 2));
    int test = int(float(count * 0.15 / 2));


    /// Shuffle the data
    system(string("../shuf-t -o ../dataset.txt ../new.txt").c_str());
    system(string("rm ../new.txt").c_str());


    /// make directory structure
    system(string("mkdir ../dataset").c_str());
    system(string("mkdir ../dataset/train").c_str());
    system(string("mkdir ../dataset/test").c_str());
    system(string("mkdir ../dataset/validation").c_str());

    /// Split the data into training and testing
    system(("split -l " + to_string(train) + " ../dataset.txt").c_str());
    system(string("mv xaa train").c_str());
    system(string("mv xab train2").c_str());
    system(("split -l " + to_string(test) + " train2").c_str());
    system(string("mv xaa validation").c_str());
    system(string("mv xab test").c_str());

/// Creating training dataset
    ifstream train_file;
    train_file.open("train");
    system(string("cp train ../dataset/train/train.txt").c_str());
    while (train_file) {
        getline(train_file, line);
        boost::split(single, line, [](char c) { return c == ' '; });
        system(("cp -v ../images/" + single[0] + " ../dataset/train").c_str());
    }
    train_file.close();

    /// Creating Validation dataset
    ifstream validation_file;
    validation_file.open("validation");
    system(string("cp validation ../dataset/validation/validation.txt").c_str());
    while(validation_file){
        getline(validation_file, line);
        boost::split(single, line, [](char c) { return c == ' '; });
        system(("cp -v ../images/" + single[0] + " ../dataset/validation/").c_str());
    }
    validation_file.close();

    /// Creating test dataset
    ifstream test_file;
    test_file.open("test");
    system(string("cp test ../dataset/test/test.txt").c_str());
    while(test_file){
        getline(test_file, line);
        boost::split(single, line, [](char c) { return c == ' '; });
        system(("cp -v ../images/" + single[0] + " ../dataset/test/").c_str());
    }
    test_file.close();

    system(string("rm ../dataset.txt").c_str());
    return 0;
}