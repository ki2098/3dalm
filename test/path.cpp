#include <iostream>
#include <filesystem>
#include <string>

using namespace std;

int main(int argc, char **argv) {
    auto file = filesystem::weakly_canonical(argv[1]);
    cout << "parent path: " << file.parent_path() << endl;
    if (file.has_parent_path())
        filesystem::current_path(file.parent_path());
    cout << filesystem::current_path() << endl;
    cout << file.filename() << endl;

    cout << filesystem::canonical(filesystem::current_path().concat("/..")) << endl;

    cout << filesystem::is_directory("foo/bar") << endl;

    cout << filesystem::path("xxx/")/"haha" << endl;
}