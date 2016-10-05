#ifndef _MODELPARAMS_H_
#define _MODELPARAMS_H_

#include "HyperParams.h"

class ModelParams
{
public:
	Alphabet _word_alpha;
	Alphabet _label_alpha;

	LookupTable _words;
	LSTM1Params _left_lstm_project; //left lstm
	LSTM1Params _right_lstm_project; //right lstm
	BiParams _activate_linear;
	UniParams _output_linear;

	SoftMaxLoss _loss;
public:
	bool initial(HyperParams& hyper_params)
	{
		if (_words.nVSize <= 0 || _label_alpha.size() <= 0)
		{
			return false;
		}
		hyper_params.wordDim = _words.nDim;
		hyper_params.labelSize = _label_alpha.size();
		hyper_params.inputSize = hyper_params.wordDim;

		_left_lstm_project.initial(hyper_params.rnnHiddenSize, hyper_params.inputSize);
		_right_lstm_project.initial(hyper_params.rnnHiddenSize, hyper_params.inputSize);
		_activate_linear.initial(hyper_params.hiddenSize, hyper_params.rnnHiddenSize, hyper_params.rnnHiddenSize, true);
		_output_linear.initial(hyper_params.labelSize, hyper_params.hiddenSize, false);
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		_words.exportAdaParams(ada);
		_left_lstm_project.exportAdaParams(ada);
		_right_lstm_project.exportAdaParams(ada);
		_activate_linear.exportAdaParams(ada);
		_output_linear.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&_words.E, "_words.E");
		checkgrad.add(&_output_linear.W, "_output_linear.W");
	}
};
#endif /*_MODELPARAMS_H_*/