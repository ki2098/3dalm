#include <string>
#include <fstream>
#include <iostream>
#include <cstdio>

using namespace std;

bool iseof(FILE *file) {
    int c = fgetc(file);
    if (c == EOF) {
        return true;
    }
    ungetc(c, file);
    return false;
}

int main() {
    string str[] = {"123", "456", "7890", "ab"};
    ofstream ofs("test", ios::binary);
    for (auto &s : str) {
        int len = s.length();
        ofs.write((char*)&len, sizeof(int));
        ofs.write((char*)s.c_str(), len*sizeof(char));
    }
    ofs.close();

    cout << "c++ style read" << endl;
    ifstream ifs("test", ios::binary);
    while (ifs.peek() != std::char_traits<char>::eof()) {
        int len;
        ifs.read((char*)&len, sizeof(int));
        string tmp(len, '\0');
        ifs.read((char*)tmp.c_str(), len*sizeof(char));
        cout << tmp << endl;
    }
    ifs.close();

    cout << "c style read" << endl;
    FILE *file = fopen("test", "rb");
    while (true) {
        if (iseof(file)) {
            break;
        }
        int len;
        fread(&len, sizeof(int), 1, file);
        char *tmp = (char*)malloc((len + 1)*sizeof(char));
        fread(tmp, sizeof(char), len, file);
        tmp[len] = '\0';
        printf("%s\n", tmp);
    }
    fclose(file);
}