#ifndef _MODELPARAMS_H_
#define _MODELPARAMS_H_

#include "HyperParams.h"

class ModelParams
{
public:
	Alphabet _word_alpha;
	Alphabet _char_alpha;
	Alphabet _label_alpha;
	LookupTable _words;
	LookupTable _chars;

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
		hyper_params.charDim = _chars.nDim;
		hyper_params.labelSize = _label_alpha.size();
		hyper_params.inputSize = hyper_params.wordDim + hyper_params.charDim; 

		hyper_params.wordWindow = hyper_params.windowContext * 2 + 1;

		_activate_linear.initial(hyper_params.hiddenSize, hyper_params.inputSize * hyper_params.wordWindow, hyper_params.inputSize * hyper_params.wordWindow, true);
		_output_linear.initial(hyper_params.labelSize, hyper_params.hiddenSize, false);
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		_words.exportAdaParams(ada);
		_chars.exportAdaParams(ada);
		_activate_linear.exportAdaParams(ada);
		_output_linear.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&_words.E, "_words.E");
		checkgrad.add(&_chars.E, "_chars.E");
		checkgrad.add(&_output_linear.W, "_output_linear.W");
	}
};
#endif /*_MODELPARAMS_H_*/