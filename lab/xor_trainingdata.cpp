#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

// Function to compute XOR output
int xor_function(int a, int b) {
    return a != b ? 1 : 0;
}

// Function to generate XOR data and save it to a file
void generate_xor_data(const string &filename, int num_samples) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file for writing." << endl;
        return;
    }

    // Write the topology line for a 2-4-1 network
    file << "topology: 2 4 1\n";

    // Seed random number generator
    srand(static_cast<unsigned>(time(0)));

    for (int i = 0; i < num_samples; ++i) {
        // Generate random inputs (0 or 1)
        int a = rand() % 2;
        int b = rand() % 2;

        // Calculate the XOR output
        int output = xor_function(a, b);

        // Write the input and output in the specified format
        file << "in: " << a << " " << b << "\n";
        file << "out: " << output << "\n";
    }

    file.close();
    cout << "Data generation completed. Saved to " << filename << endl;
}

int main(int argc, char *argv[]) {
    int num_samples = 5000; // Default number of XOR samples

    // Check if the user provided a filename as a command-line argument
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <filename>" << endl;
        return 1;
    }

    string filename = argv[1];
    generate_xor_data(filename, num_samples);

    return 0;
}
