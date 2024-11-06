#include <vector> // include header so that referencing is possible
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cassert>

using namespace std; // defines that standard is in std -> therefore no std:: required

// class for reading training data
class TrainingData{

public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology) {
    string line;
    string label;
    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }
    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
    return;
}

TrainingData::TrainingData(const string filename) {
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals) {
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals) {
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}

struct Connection { // defines connections between neuron, including weight and also the delta of the weight
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer; // define that layer is made up of vectors with neurons

// ************** class Neuron *****************

class Neuron {

public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta; // [0.0..1.0] lr
    static double alpha; // [0.0..n] momentum
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); } // function for random weight initialization
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights; // defines weights and deltas
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.1;
double Neuron::alpha = 0.3;

void Neuron::updateInputWeights(Layer &prevLayer) {
    // weights to be updated in connection container
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight =
            eta // overall learning rate
            * neuron.getOutputVal()
            * m_gradient
            + alpha // momentum
            * oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;
    // sum individual "path contributions" of errors
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x) {
    return tanh(x); // simply from math
}

double Neuron::transferFunctionDerivative(double x) {
    return 1.0 - x * x; // approximation of tanh derivative (very fast)
}

void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0.0; // define sum (sum up previous layers outputs which are our inputs) + bias
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
            prevLayer[n].m_outputWeights[m_myIndex].weight; // all inputs * connection weights
    }
    m_outputVal = transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) { // defines what neuron does (constructor)
    for (unsigned c = 0; c < numOutputs; ++c) { // add connections for all neuron connections
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight(); // define the weight of the neuron with random number function
    }
    m_myIndex = myIndex;
}

// ************** class Net *****************

class Net {
public:
    Net(const vector<unsigned> &topology); // define net with topology
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum], able to reference neurons and layers
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor = 100.0;
};

void Net::getResults(vector<double> &resultVals) const {
    resultVals.clear();
    // Exclude the last neuron if it's a bias neuron
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}


void Net::backProp(const vector<double> &targetVals) {
    // Calculate overall net error (loss)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average squared error
    m_error = sqrt(m_error); // rms

    // Implement a recent average measurement
    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
        / (m_recentAverageSmoothingFactor + 1.0);

    // Output layer gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Gradients in hidden layers
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers, update connection weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const vector<double> &inputVals) {
    assert(inputVals.size() == m_layers[0].size() - 1); // error checking

    // Assign (latch) input values to input neurons
    for (unsigned i = 0; i < inputVals.size(); i++) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagation
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const vector<unsigned> &topology) { // define what Net does
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // Made new layer, now fill it with neurons and add bias
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a Neuron!" << endl;
        }

        // Force bias node's output to 1.0
        m_layers.back().back().setOutputVal(1.0);
    }
}

void showVectorVals(string label, vector<double> &v) {
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }
    cout << endl;
}

// ************** Main *****************

int main() {
    TrainingData trainData("/home/luki/NN_scratch_cpp/lab/xorTrainingData.txt");

    vector<unsigned> topology;
    trainData.getTopology(topology);
    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;
    vector<double> errorLog; // Vektor zum Speichern der Fehlerwerte

    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual results
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net with the expected outputs
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Speichere den aktuellen Fehler im Vektor
        double error = myNet.getRecentAverageError();
        errorLog.push_back(error);

        // Report how well the training is working
        cout << "Net recent average error: " << error << endl;
    }

    cout << endl << "Done" << endl;

    // Schreibe die Fehlerwerte in eine Datei
    ofstream outFile("errorLog.txt");
    if (outFile.is_open()) {
        for (unsigned i = 0; i < errorLog.size(); ++i) {
            outFile << i + 1 << " " << errorLog[i] << std::endl;
        }
        outFile.close();
        cout << "Fehlerwerte erfolgreich in errorLog.txt gespeichert." << endl;
    } else {
        cerr << "Fehler beim Öffnen der Datei!" << endl;
    }

    return 0;
}
