#ifndef _COMPUTIONGRAPH_H_
#define _COMPUTIONGRAPH_H_

#include "ModelParams.h"
#include "Example.h"

class ComputionGraph:public Graph
{
public:
	const static int max_sentence_size = 128;
	vector<LookupNode> _target_word_inputs;
	vector<LookupNode> _tweet_word_inputs;
	WindowBuilder _target_window;
	WindowBuilder _tweet_window;
	MaxPoolNode _max_target_pooling;
	MaxPoolNode _max_tweet_pooling;
	BiNode _activate;

	LinearNode _output;

public:
	inline void createNodes(int sent_length)
	{
		_target_word_inputs.resize(sent_length);
		_tweet_word_inputs.resize(sent_length);
		_target_window.resize(sent_length);
		_tweet_window.resize(sent_length);
	}

	inline void clear()
	{
		Graph::clear();
		_target_word_inputs.clear();
		_tweet_word_inputs.clear();
		_target_window.clear();
		_tweet_window.clear();
	}

	inline void initial(ModelParams& model_params, HyperParams& hyper_params)
	{
		for (int i = 0; i < _target_word_inputs.size(); i++)
		{
			_target_word_inputs[i].setParam(&model_params._words);
			_tweet_word_inputs[i].setParam(&model_params._words);
		}
		_target_window.setContext(hyper_params.windowContext);
		_tweet_window.setContext(hyper_params.windowContext);
		_activate.setParam(&model_params._activate_linear);
		_output.setParam(&model_params._output_linear);
	}
	
	inline void forward(const Feature& feature, bool bTrain = false)
	{
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation
		int max_target_input_length, max_tweet_input_length;
		feature.m_target_words.size() > max_sentence_size ? max_target_input_length = max_sentence_size : max_target_input_length = feature.m_target_words.size();
		feature.m_tweet_words.size() > max_sentence_size ? max_tweet_input_length = max_sentence_size : max_tweet_input_length = feature.m_tweet_words.size();

		for (int i = 0; i < max_target_input_length; i++)
			_target_word_inputs[i].forward(this, feature.m_target_words[i]);
		for (int i = 0; i < max_tweet_input_length; i++)
			_tweet_word_inputs[i].forward(this, feature.m_tweet_words[i]);
		_target_window.forward(this, getPNodes(_target_word_inputs, max_target_input_length));

		_tweet_window.forward(this, getPNodes(_tweet_word_inputs, max_tweet_input_length));

		_max_target_pooling.forward(this, getPNodes(_target_window._outputs, max_target_input_length));
		_max_tweet_pooling.forward(this, getPNodes(_tweet_window._outputs, max_tweet_input_length));
		_activate.forward(this, &_max_target_pooling, &_max_tweet_pooling);
		_output.forward(this, &_activate);
	}
};

#endif /*_COMPUTIONGRAPH_H_*/