#include <iostream>
#include "../src/argparse.hpp"

using namespace std;
using namespace argparse;

int main(int argc, char **argv) {
    ArgumentParser parser;
    parser.add_argument("--clear", "-c")
        .help("clear previous files.")
        .default_value(false)
        .implicit_value(true)
        .required();
    
    parser.parse_args(argc, argv);
    if (parser["-c"] == true) {
        cout << "clear previous files." << endl;
    }
}