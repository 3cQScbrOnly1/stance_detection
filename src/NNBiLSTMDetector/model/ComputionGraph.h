#ifndef _COMPUTIONGRAPH_H_
#define _COMPUTIONGRAPH_H_

#include "ModelParams.h"
#include "Example.h"

class ComputionGraph:public Graph
{
public:
	const static int max_sentence_size = 256;
	vector<LookupNode> _word_inputs;
	LSTM1Builder _left_lstm;
	LSTM1Builder _right_lstm;
	BiNode _activate;
	LinearNode _output;

public:
	inline void createNodes(int sent_length)
	{
		_word_inputs.resize(sent_length);
		_left_lstm.resize(sent_length);
		_right_lstm.resize(sent_length);
	}

	inline void clear()
	{
		Graph::clear();
		_left_lstm.clear();
		_right_lstm.clear();
	}

	inline void initial(ModelParams& model_params, HyperParams& hyper_params)
	{
		for (int i = 0; i < _word_inputs.size(); i++)
			_word_inputs[i].setParam(&model_params._words);
		_left_lstm.setParam(&model_params._left_lstm_project, hyper_params.dropProb, true);
		_right_lstm.setParam(&model_params._right_lstm_project, hyper_params.dropProb, false);
		_activate.setParam(&model_params._activate_linear);
		_output.setParam(&model_params._output_linear);
	}
	
	inline void forward(const Feature& feature, bool bTrain = false)
	{
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation
		int target_size, tweet_size;
		feature.m_target_words.size() > max_sentence_size ? target_size = max_sentence_size : target_size = feature.m_target_words.size();
		feature.m_tweet_words.size() > max_sentence_size ? tweet_size = max_sentence_size : tweet_size = feature.m_tweet_words.size();

		for (int i = 0; i < target_size; i++)
			_word_inputs[i].forward(this, feature.m_target_words[i]);
		for (int i = 0; i < tweet_size; i++)
			_word_inputs[i + target_size].forward(this, feature.m_tweet_words[i]);
		_left_lstm.forward(this, getPNodes(_word_inputs, target_size + tweet_size));
		_right_lstm.forward(this, getPNodes(_word_inputs, target_size + tweet_size));

		_activate.forward(this, &_left_lstm._hiddens_drop[target_size], &_right_lstm._hiddens_drop[target_size + tweet_size - 1]);
		_output.forward(this, &_activate);
	}
};

#endif /*_COMPUTIONGRAPH_H_*/