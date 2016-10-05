#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include <vector>

using namespace std;

class Feature
{
public:
	vector<string> m_target_words;
	vector<string> m_tweet_words;
	vector<vector<string> > m_target_chars;
	vector<vector<string> > m_tweet_chars;
public:
	void clear()
	{
		m_target_words.clear();
		m_tweet_words.clear();
		m_target_chars.clear();
		m_tweet_chars.clear();
	}
};

class Example
{
public:
	Feature m_feature;
	vector<double> m_label;

public:
	void clear()
	{
		m_feature.clear();
		m_label.clear();
	}
};

#endif /*_EXAMPLE_H_*/