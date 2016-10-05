#ifndef _INSTANCE_H_
#define _INSTANCE_H_

#include <iostream>
using namespace std;

class Instance
{
public:
	void clear()
	{
		m_id.clear();
		m_target.clear();
		m_tweet.clear();
		m_stance.clear();
	}

	void evaluate(const string& predict_label, Metric& eval) const
	{
		if (predict_label == m_stance)
			eval.correct_label_count++;
		eval.overall_label_count++;
	}

	void evaluate(const string& predict_label, Metric& against, Metric& favor)
	{
		if (predict_label == m_stance)
		{
			if (m_stance == "FAVOR")
			{
				favor.correct_label_count++;
				favor.predicated_label_count++;
			}
			if (m_stance == "AGAINST")
			{
				against.correct_label_count++;
				against.predicated_label_count++;
			}
		}
		else
		{
			if (predict_label == "FAVOR")
				favor.predicated_label_count++;
			if (predict_label == "AGAINST")
				against.predicated_label_count++;
		}
		if (m_stance == "FAVOR")
			favor.overall_label_count++;
		if (m_stance == "AGAINST")
			against.overall_label_count++;
	}

public:
	string m_id;
	vector<string> m_target;
	vector<string> m_tweet;
	string m_stance;
};

#endif /*_INSTANCE_H_*/
