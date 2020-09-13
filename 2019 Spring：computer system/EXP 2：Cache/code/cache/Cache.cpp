#include <iostream>
#include "Cache.h"
#include "type.h"
using namespace std;

Cache::Cache(size_t s, size_t ls, size_t ws, time_t cal, time_t mal, time_t cpuct)
	: S{ s }, LS{ ls }, WS{ ws }, CAL{ cal }, MAL{ mal }, CPUCT{ cpuct }
{
	// LN = S * 1024 / LS;
	LN = S / LS;
	LWN = LS / WS;

	memory = new Line[LN + 1];
	for (int i = 0; i < LN; i++) {
		memory[i].valid = false;
		memory[i].count = 0;
	}

	HC = 0; MC = 0; TABC = 0;
	RC = 0; WC = 0;
	HR = 0; MR = 0;
	RTT = 0; WTT = 0; AAT = 0;
	RAT = 0; WAT = 0; TP = 0;
	SUR = 0; SU = 0;

}

Cache::~Cache()
{
	delete[] memory;
}

void Cache::read(address_t addr)
{
	Address d_addr;
	decode(addr, d_addr);
	int4_t index;
	if (isInCache(d_addr, index)) {
		HC++;
		TABC += WS;
		RTT += CAL;
	}
	else {
		handleReadMiss(d_addr);
		MC++;
		TABC += WS;
		RTT += (MAL + CAL);
	}
	RC++;
}

void Cache::write(address_t addr)
{
	Address d_addr;
	decode(addr, d_addr);
	int4_t index;
	if (isInCache(d_addr, index)) {
		HC++;
		TABC += WS;
		memory[index].dirty = true;
		if (writeStrategy == WBack) WTT += CAL;
		// modify cache only in writeStrategy
		else WTT += (MAL + CAL);
		// modify cache and memory simutaniously in writethrough strategy
	} 
	else {
		handleWriteMiss(d_addr);
		MC++;
		TABC += WS;
		if (writeStrategy == WBack) WTT += (MAL + CAL); 
		// load it into cache from memory and modify it
		else WTT += (MAL + MAL + CAL); 
		// write data into memory and load it into cache
	}
	WC++;
}

void Cache::printInfo()
{
	Cache::caculateInfo();
	cout << "Cache parameters: " << endl;
	cout << "Cache Size: " << S << "B,\t" << "Line Size: " << LS << "B,\t"
		<< "Word Size: " << WS << "B" << endl;
	cout << "Total number of lines: " << LN << ",\t"
		<< "Toal number of word in a line: " << LWN << endl;
	cout << "Cache Access Latency: " << CAL << "ns,\t"
		<< "Memory Access Latency: " << MAL << "ns,\t" 
		<< "CPU Cycle: " << CPUCT << "ns" << endl;
	cout << "Performance parameters: " << endl;
	cout << "Hit Ratio: " << HR * 100 << "%,\t"
		<< "Miss Ratio: " << MR * 100 << "%" << endl;
	cout << "Average Read Time: " << RAT << "ns\t"
		<< "Average Write Time: " << WAT << "ns\t"
		<< "Average Access Time: " << AAT << "ns\t" << endl;
	cout << "Throughput: " << TP << "MB/s" << endl;
	cout << "Speed Up Ratio: " << SUR * 100 << "%" << endl;
	cout << "Space Utilization: " << SU * 100 << "%" << endl;

}

void Cache::caculateInfo()
{
	HR = 1.0 * HC / (HC + MC);
	MR = 1.0 * MC / (HC + MC);
	if (RC != 0) RAT = 1.0 * RTT / RC;
	else RAT = 0;
	if (WC != 0) WAT = 1.0 * WTT / WC;
	else WAT = 0;
	AAT = 1.0 * (RTT + WTT) / (RC + WC);
	TP = 1.0 * TABC / (RTT + WTT) * 1000 * 1000 * 1000 / 1024 / 1024;
	SUR = 1.0 * ((RC + WC) * (MAL + CAL) - (RTT + WTT)) / ((RC + WC) * (MAL + CAL));
	for (int i = 0; i < LN; i++) {
		if (memory[i].valid == true) SU++;
	}
	SU = SU / LN;
}

void Cache::clear()
{
	HC = 0; MC = 0; TABC = 0;
	RC = 0; WC = 0;
	HR = 0; MR = 0;
	RTT = 0; WTT = 0; AAT = 0;
	RAT = 0; WAT = 0; TP = 0;
	SUR = 0; SU = 0;
}
