#ifndef _DRIVER_H_
#define _DRIVER_H_

#include <iostream>
#include "ComputionGraph.h"

class Driver
{
public:
	Driver() {
		_pcg = NULL;
	}

	~Driver() {
		if (_pcg != NULL)
			delete _pcg;
	}
public:
	ComputionGraph *_pcg;  // build neural graphs
	ModelParams _model_params;  // model parameters
	HyperParams _hyper_params;

	Metric _eval;
	CheckGrad _checkgrad;
	ModelUpdate _ada;  // model update
public:
	inline void initial()
	{
		if (!_hyper_params.bVaild()){
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}

		if (!_model_params.initial(_hyper_params)){
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_model_params.exportModelParams(_ada);
		_model_params.exportCheckGradParams(_checkgrad);
		_hyper_params.print();

		_pcg = new ComputionGraph();
		_pcg->createNodes(ComputionGraph::max_sentence_size);
		_pcg->initial(_model_params, _hyper_params);

		setUpdateParameters(_hyper_params.nnRegular, _hyper_params.adaAlpha, _hyper_params.adaEps);

	}

	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps)
	{
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

	inline dtype train(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			//forward
			_pcg->forward(example.m_feature, true);

			//loss function
			//for (int idx = 0; idx < seq_size; idx++) {
			//cost += _loss.loss(&(_pcg->output[idx]), example.m_labels[idx], _eval, example_num);				
			//}
			cost += _model_params._loss.loss(&_pcg->_output, example.m_label, _eval, example_num);

			// backward, which exists only for training 
			_pcg->backward();
		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	inline void predict(const Feature& feature, int& result) {
		_pcg->forward(feature);
		//results.resize(seq_size);
		//for (int idx = 0; idx < seq_size; idx++) {
		//	_loss.predict( &(_pcg->output[idx]), results[idx]);
		//}
		_model_params._loss.predict(&_pcg->_output, result);
	}

	inline dtype cost(const Example& example){
		_pcg->forward(example.m_feature); //forward here

		dtype cost = 0.0;
		//loss function
		//for (int idx = 0; idx < seq_size; idx++) {
		//	cost += _loss.cost(&(_pcg->output[idx]), example.m_labels[idx], 1);
		//}
		cost += _model_params._loss.cost(&_pcg->_output, example.m_label, 1);

		return cost;
	}


	void updateModel() {
		//_ada.update();
		_ada.update(5.0);
	}

	void checkgrad(const vector<Example>& examples, int iter){
		ostringstream out;
		out << "Iteration: " << iter;
		_checkgrad.check(this, examples, out.str());
	}

};
#endif /*_DRIVER_H_*/