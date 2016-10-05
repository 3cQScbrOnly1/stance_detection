#ifndef _COMPUTIONGRAPH_H_
#define _COMPUTIONGRAPH_H_

#include "ModelParams.h"
#include "Example.h"

class ComputionGraph:public Graph
{
public:
	const static int max_sentence_size = 128;
	const static int max_char_size = 64;
	vector<LookupNode> _target_word_inputs;
	vector<LookupNode> _tweet_word_inputs;

	vector<vector<LookupNode> > _target_char_inputs;
	vector<vector<LookupNode> > _tweet_char_inputs;

	vector<MaxPoolNode> _max_poolings_target_char;
	vector<MaxPoolNode> _max_poolings_tweet_char;

	vector<ConcatNode> _target_word_char_concat;
	vector<ConcatNode> _tweet_word_char_concat;

	WindowBuilder _target_window;
	WindowBuilder _tweet_window;

	MaxPoolNode _max_target_pooling;
	MaxPoolNode _max_tweet_pooling;
	BiNode _activate;

	LinearNode _output;

public:
	inline void createNodes(int sent_length, int char_length)
	{
		_target_word_inputs.resize(sent_length);
		_tweet_word_inputs.resize(sent_length);

		_target_char_inputs.resize(sent_length);
		for (int i = 0; i < sent_length; i++)
			_target_char_inputs[i].resize(char_length);

		_tweet_char_inputs.resize(sent_length);
		for (int i = 0; i < sent_length; i++)
			_tweet_char_inputs[i].resize(char_length);

		_max_poolings_target_char.resize(sent_length);
		_max_poolings_tweet_char.resize(sent_length);
		_target_word_char_concat.resize(sent_length);
		_tweet_word_char_concat.resize(sent_length);

		_target_window.resize(sent_length);
		_tweet_window.resize(sent_length);
	}

	inline void clear()
	{
		Graph::clear();
		_target_word_inputs.clear();
		_tweet_word_inputs.clear();
		_target_char_inputs.clear();
		_tweet_char_inputs.clear();

		_target_word_char_concat.clear();
		_tweet_word_char_concat.clear();

		_target_window.clear();
		_tweet_window.clear();
	}

	inline void initial(ModelParams& model_params, HyperParams& hyper_params)
	{
		for (int i = 0; i < _target_word_inputs.size(); i++)
		{
			_target_word_inputs[i].setParam(&model_params._words);
			for (int j = 0; j < _target_char_inputs[i].size(); j++)
				_target_char_inputs[i][j].setParam(&model_params._chars);
		}
		for (int i = 0; i < _tweet_word_inputs.size(); i++)
		{
			_tweet_word_inputs[i].setParam(&model_params._words);
			for (int j = 0; j < _tweet_char_inputs[i].size(); j++)
				_tweet_char_inputs[i][j].setParam(&model_params._chars);
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
		int target_size, tweet_size, char_target_size, char_tweet_size;
		feature.m_target_words.size() > max_sentence_size ? target_size = max_sentence_size : target_size = feature.m_target_words.size();
		feature.m_tweet_words.size() > max_sentence_size ? tweet_size = max_sentence_size : tweet_size = feature.m_tweet_words.size();

		for (int i = 0; i < target_size; i++)
		{
			_target_word_inputs[i].forward(this, feature.m_target_words[i]);
			feature.m_target_chars[i].size() > max_char_size ? char_target_size = max_char_size : char_target_size = feature.m_target_chars[i].size();
			for (int j = 0; j < char_target_size; j++)
			{
				_target_char_inputs[i][j].forward(this, feature.m_target_chars[i][j]);
			}
			_max_poolings_target_char[i].forward(this, getPNodes(_target_char_inputs[i], char_target_size));
			_target_word_char_concat[i].forward(this, &_target_word_inputs[i], &_max_poolings_target_char[i]);
		}

		for (int i = 0; i < tweet_size; i++)
		{
			_tweet_word_inputs[i].forward(this, feature.m_tweet_words[i]);
			feature.m_tweet_chars[i].size() > max_char_size ? char_tweet_size = max_char_size : char_tweet_size = feature.m_tweet_chars[i].size();
			for (int j = 0; j < char_tweet_size; j++)
			{
				_tweet_char_inputs[i][j].forward(this, feature.m_tweet_chars[i][j]);
			}
			_max_poolings_tweet_char[i].forward(this, getPNodes(_tweet_char_inputs[i], char_tweet_size));
			_tweet_word_char_concat[i].forward(this, &_tweet_word_inputs[i], &_max_poolings_tweet_char[i]);
		}

		_target_window.forward(this, getPNodes(_target_word_char_concat, target_size));
		_tweet_window.forward(this, getPNodes(_tweet_word_char_concat, tweet_size));

		_max_target_pooling.forward(this, getPNodes(_target_window._outputs, target_size));
		_max_tweet_pooling.forward(this, getPNodes(_tweet_window._outputs, tweet_size));
		_activate.forward(this, &_max_target_pooling, &_max_tweet_pooling);
		_output.forward(this, &_activate);
	}
};

#endif /*_COMPUTIONGRAPH_H_*/