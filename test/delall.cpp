#include <filesystem>

using namespace std;

int main() {
    filesystem::path curpath = filesystem::current_path();
    filesystem::remove_all(curpath);
    filesystem::create_directories(curpath);
}