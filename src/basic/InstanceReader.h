#ifndef _INSTANCEREADER_H_
#define _INSTANCEREADER_H_

#include <vector>
#include <fstream>
#include <iostream>

#include "Instance.h"
#include "N3L.h"

using namespace std;

class InstanceReader
{
public:
	void load(string path, vector<Instance>& instances, int maxInstance = -1)
	{
		ifstream file(path.c_str());
		string line;
		vector<string> data, target, tweet;
		Instance ins;
		int count = 0;
		while (getline(file, line))
		{
			split_bychars(line, data, "\t");
			if (data.size() == 4)
			{
				split_bychar(data[1], target, ' ');
				split_bychar(data[2], tweet, ' ');
				ins.m_id = data[0];
				ins.m_target = target;
				ins.m_tweet = tweet;
				ins.m_stance = data[3];
				instances.push_back(ins);
				count++;
			}
			if (count == maxInstance)
				break;
		}
		file.close();
	}
};

#endif /*_INSTANCEREADER_H_*/