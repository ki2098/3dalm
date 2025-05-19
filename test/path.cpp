#include <iostream>
#include <filesystem>
#include <string>

using namespace std;

int main(int argc, char **argv) {
    auto file = filesystem::absolute(argv[1]);
    cout << "parent path: " << file.parent_path() << endl;
    filesystem::current_path(file.parent_path());
    cout << filesystem::current_path() << endl;

    cout << filesystem::canonical(filesystem::current_path().concat("/..")) << endl;
}