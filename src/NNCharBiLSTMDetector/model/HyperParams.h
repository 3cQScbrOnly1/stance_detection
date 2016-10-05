#ifndef _HYPERPARAMS_H_
#define _HYPERPARAMS_H_

#include "N3L.h"
#include "Options.h"

struct HyperParams
{
public:
	int hiddenSize;
	int rnnHiddenSize;
	dtype dropProb;

	int inputSize;
	int labelSize;
	int wordDim;
	int charDim;
	dtype nnRegular, adaAlpha, adaEps; // for optimization

public:
	HyperParams(){
		bAssigned = false;
	}

	void setRequared(Options& opt)
	{
		hiddenSize = opt.hiddenSize;
		rnnHiddenSize = opt.rnnHiddenSize;
		dropProb = opt.dropProb;

		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;

		bAssigned = true;
	}
	
	void clear()
	{
		bAssigned = false;
	}

	void print()
	{
		cout << "hyper parameters " << endl;
		cout << "hiddenSize=" << hiddenSize << endl;
		cout << "rnnHiddenSize=" << rnnHiddenSize << endl;
		cout << "dropProb=" << dropProb << endl;
		cout << "nnRegular=" << nnRegular << endl;
		cout << "adaAlpha=" << adaAlpha << endl;
		cout << "adaEps=" << adaEps << endl;
	}

	bool bVaild()
	{
		return bAssigned;
	}

private:
	bool bAssigned;
};

#endif /*_HYPERPARAMS_H_*/