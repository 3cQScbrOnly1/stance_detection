#ifndef _NNBILSTMDETECTOR_H_
#define _NNBILSTMDETECTOR_H_

#include "Options.h"
#include "Instance.h"
#include "Driver.h"
#include "Example.h"
#include "InstanceReader.h"
#include "N3L.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Detector
{
public:
	Options m_options;
	unordered_map<string, int> m_word_stats;
	Driver m_driver;

public:
	void createAlphabet(const vector<Instance>& train_instances);
	void addTestAlphabet(const vector<Instance>& test_instances);
	void initialExamples(const vector<Instance>& instances, vector<Example>& examples);
	void extractFeature(const Instance& instance, Feature& feature);
	void convert2Example(const Instance& instance, Example& example);

	void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile);
	void predict(const Feature& feature, string& output);
};

#endif