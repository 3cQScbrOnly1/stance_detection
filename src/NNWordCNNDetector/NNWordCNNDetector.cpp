#include "Argument_helper.h"
#include "NNWordCNNDetector.h"

void Detector::createAlphabet(const vector<Instance>& train_instances)
{
	if (train_instances.size() == 0)
	{
		std::cout << "Training set is empty. " << std::endl;
	}
	else
	{
		std::cout << "Creating alphabet." << std::endl;
		m_driver._model_params._label_alpha.clear();
		string cur_word;
		for (int i = 0; i < train_instances.size(); i++)
		{
			
			m_driver._model_params._label_alpha.from_string(train_instances[i].m_stance);
			for (int j = 0; j < train_instances[i].m_target.size(); j++)
			{
				cur_word = normalize_to_lowerwithdigit(train_instances[i].m_target[j]);
				m_word_stats[cur_word]++;
			}
			for (int j = 0; j < train_instances[i].m_tweet.size(); j++)
			{
				cur_word = normalize_to_lowerwithdigit(train_instances[i].m_tweet[j]);
				m_word_stats[cur_word]++;
			}
		}
		cout << "label num: " << m_driver._model_params._label_alpha.size() << endl;
	}
}

void Detector::addTestAlphabet(const vector<Instance>& test_instances)
{
	cout << "add test alpha" << endl;
	string cur_word;
	int target_size, tweet_size;
	int ins_size = test_instances.size();
	for (int i = 0; i < ins_size; i++)
	{

		m_driver._model_params._label_alpha.from_string(test_instances[i].m_stance);
		target_size = test_instances[i].m_target.size();
		for (int j = 0; j < target_size; j++)
		{
			cur_word = normalize_to_lowerwithdigit(test_instances[i].m_target[j]);
			if (!m_options.wordEmbFineTune)
				m_word_stats[cur_word]++;
		}
		tweet_size = test_instances[i].m_tweet.size();
		for (int j = 0; j < tweet_size; j++)
		{
			if (!m_options.wordEmbFineTune)
				cur_word = normalize_to_lowerwithdigit(test_instances[i].m_tweet[j]);
			m_word_stats[cur_word]++;
		}
	}
}

void Detector::extractFeature(const Instance& instance, Feature& feature)
{
	feature.clear();
	feature.m_tweet_words = instance.m_tweet;
	feature.m_target_words = instance.m_target;
	/*
	int tweet_size = instance.m_tweet.size();
	int target_size = instance.m_target.size();
	vector<string> chs;
	for (int i = 0; i < tweet_size; i++)
	{
		getCharactersFromString(instance.m_tweet[i], chs);
		feature.m_tweet_chars.push_back(chs);
	}
	for (int i = 0; i < target_size; i++)
	{
		getCharactersFromString(instance.m_target[i], chs);
		feature.m_target_chars.push_back(chs);
	}
	*/
}

void Detector::convert2Example(const Instance& instance, Example& example)
{
	example.clear();
	int label_size = m_driver._model_params._label_alpha.size();
	int id;
	string str_label, ins_label = instance.m_stance;
	for (int i = 0; i < label_size; i++)
	{
		str_label = m_driver._model_params._label_alpha.from_id(i);
		if (str_label == ins_label)
			example.m_label.push_back(1.0);
		else
			example.m_label.push_back(0.0);
	}
	extractFeature(instance, example.m_feature);
}

void Detector::initialExamples(const vector<Instance>& instances, vector<Example>& examples)
{
	int ins_size = instances.size();
	Example example;
	for (int i = 0; i < ins_size; i++)
	{
		convert2Example(instances[i], example);
		examples.push_back(example);
	}
}

void Detector::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile)
{
	if (optionFile != "")
		m_options.load(optionFile);
	m_options.showOptions();
	vector<Instance> train_instances, dev_instances, test_instances;
	InstanceReader the_reader;
	the_reader.load(trainFile, train_instances, m_options.maxInstance);
	if (devFile != "")
		the_reader.load(devFile, dev_instances, m_options.maxInstance);
	if (testFile != "")
		the_reader.load(testFile, test_instances);
	cout << "training set size:" << train_instances.size() << endl; 
	cout << "dev set size: " << dev_instances.size() << endl;
	cout << "test set size: " << test_instances.size() << endl;
	createAlphabet(train_instances);
	addTestAlphabet(dev_instances);
	addTestAlphabet(test_instances);

	vector<Example> train_examples, dev_examples, test_examples;
	initialExamples(train_instances, train_examples);
	initialExamples(dev_instances, dev_examples);
	initialExamples(test_instances, test_examples);
	m_word_stats[unknownkey] = m_options.wordCutOff + 1;
	m_driver._model_params._word_alpha.initial(m_word_stats, m_options.wordCutOff);
	if (m_options.wordFile != "") {
		m_driver._model_params._words.initial(&m_driver._model_params._word_alpha, m_options.wordFile, m_options.wordEmbFineTune);
	}
	else{
		m_driver._model_params._words.initial(&m_driver._model_params._word_alpha, m_options.wordEmbSize, m_options.wordEmbFineTune);
	}
	m_driver._hyper_params.setRequared(m_options);
	m_driver.initial();
	
	int inputSize = train_examples.size();

	int batchBlock = inputSize / m_options.batchSize;
	if (inputSize % m_options.batchSize != 0)
		batchBlock++;

	srand(0);
	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);

	static Metric eval, metric_dev, metric_test, metric_dev_against, metric_dev_favor;
	static vector<Example> subExamples;
	dtype bestDIS = 0;
	int devNum = dev_examples.size(), testNum = test_examples.size();
	for (int iter = 0; iter < m_options.maxIter; ++iter) 
	{
		std::cout << "##### Iteration " << iter << std::endl;
		random_shuffle(indexes.begin(), indexes.end());
		eval.reset();
		for (int updateIter = 0; updateIter < batchBlock; updateIter++) 
		{
			subExamples.clear();
			int start_pos = updateIter * m_options.batchSize;
			int end_pos = (updateIter + 1) * m_options.batchSize;
			if (end_pos > inputSize)
				end_pos = inputSize;

			for (int idy = start_pos; idy < end_pos; idy++) 
				subExamples.push_back(train_examples[indexes[idy]]);
			int curUpdateIter = iter * batchBlock + updateIter;
			dtype cost = m_driver.train(subExamples, curUpdateIter);

			eval.overall_label_count += m_driver._eval.overall_label_count;
			eval.correct_label_count += m_driver._eval.correct_label_count;

			if ((curUpdateIter + 1) % m_options.verboseIter == 0) 
			{
				//m_driver.checkgrad(subExamples, curUpdateIter + 1);
				std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
				std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
			}
			m_driver.updateModel();
		}

		bool current_better = false;
		if (devNum > 0)
		{
			string predict_label;
			metric_dev.reset();
			metric_dev_against.reset();
			metric_dev_favor.reset();
			for (int i = 0; i < devNum; i++)
			{
				predict(dev_examples[i].m_feature, predict_label);
				//dev_instances[i].evaluate(predict_label, metric_dev);
				dev_instances[i].evaluate(predict_label, metric_dev_against, metric_dev_favor);
			}
			cout << "against:";
			metric_dev_against.print();
			cout << "favor:";
			metric_dev_favor.print();
			//metric_dev.print();
			if (metric_dev.getAccuracy() > bestDIS)
				current_better = true;
		}
	}
}

void Detector::predict(const Feature& feature, string& output)
{
	int label_idx;
	m_driver.predict(feature, label_idx);
	output = m_driver._model_params._label_alpha.from_id(label_idx, unknownkey);

	if (output == nullkey){
		cout << "predict error" << endl;
	}
}

int main(int argc, char* argv[]) {

	std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	dsr::Argument_helper ah;


	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

	ah.process(argc, argv);
	Detector the_detector;
	the_detector.train(trainFile, devFile, testFile, modelFile, optionFile);
}