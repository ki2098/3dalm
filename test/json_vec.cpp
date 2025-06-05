#include <vector>
#include <iostream>
#include "../src/json.hpp"

using json = nlohmann::json;
using namespace std;

int main() {
    string json_str = "{\"data\":[1,2,3,4,5]}";
    auto &&data_json = json::parse(json_str);
    vector<int> data = data_json["data"].get<vector<int>>();
    for (auto &e : data) {
        cout << e << endl;
    }
}