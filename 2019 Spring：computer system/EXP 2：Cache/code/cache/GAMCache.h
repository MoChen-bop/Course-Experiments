#pragma once
#include <iostream>
#include <stdlib.h>
#include "Cache.h"

using namespace std;

class GAMCache : public Cache
{
public:
	GAMCache(size_t s, size_t ls, size_t ws, time_t cal, 
		time_t mal, time_t cpuct, int4_t gs);
	void read(address_t addr);
	void write(address_t addr);
	void printInfo();
protected:

	bool isInCache(Address d_addr, int4_t& index);
	void decode(address_t addr, Address& d_addr);
	void handleReadMiss(const Address& d_addr);
	void handleWriteMiss(const Address& d_addr);
	void RReplace(const Address& d_addr, bit_t w);
	void LFUReplace(const Address& d_addr, bit_t w);
	void LRUReplace(const Address& d_addr, bit_t w);
	void FIFOReplace(const Address& d_addr, bit_t w);

	int4_t SN;  // number of set in cache
	int4_t GS;  // number of lines in one group
	int4_t OBN; // number of bit in offset 
	int4_t SBN; // number of bit in cache set

	int4_t* lastLoadIndex; // 1D array of index in which data was loaded lately
	                       // used in FIFO replacement strategy
};

GAMCache::GAMCache(size_t s, size_t ls, size_t ws, 
	time_t cal, time_t mal, time_t cpuct, int4_t gs)
	: Cache{ s, ls, ws, cal, mal, cpuct }
{
	GS = gs;
	OBN = 0;
	int4_t tmp = LS;
	while (tmp != 1) {
		OBN++;
		tmp /= 2;
	}

	SN = LN / GS;

	SBN = 0;
	tmp = SN;
	while (tmp != 1) {
		SBN++;
		tmp /= 2;
	}

	lastLoadIndex = new int4_t[SN];
	for (int i = 0; i < SN; i++) {
		lastLoadIndex[i] = -1;
	}
}

inline void GAMCache::read(address_t addr)
{
	Address d_addr;
	decode(addr, d_addr);
	int4_t index;
	if (isInCache(d_addr, index)) {
		HC++;
		TABC += WS;
		RTT += CAL;

		if (replaceStrategy == LFUR) memory[index].count++;
		else if (replaceStrategy == LRUR) {
			int baseIndex = d_addr.set * GS;
			for (int i = baseIndex; i < baseIndex + GS; i++) {
				if (i != index) memory[i].count++;
			}
		}
	}
	else {
		handleReadMiss(d_addr);
		MC++;
		TABC += WS;
		RTT += (MAL + CAL);
	}
	RC++;
}

inline void GAMCache::write(address_t addr)
{
	Address d_addr;
	decode(addr, d_addr);
	int4_t index;
	if (isInCache(d_addr, index)) {
		HC++;
		TABC += WS;
		WTT += CAL;
		memory[index].dirty = true;

		if (replaceStrategy == LFUR) memory[index].count++;
		else if (replaceStrategy == LRUR) {
			int baseIndex = d_addr.set * GS;
			for (int i = baseIndex; i < baseIndex + GS; i++) {
				if (i != index) memory[i].count++;
			}
		}
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

inline void GAMCache::printInfo()
{
	cout << "----------Group Associated Mapping Cache----------" << endl;
	if (writeStrategy == WBack) cout << "Using Write Back Strategy." << endl;
	else cout << "Using Write Through Strategy." << endl;

	if (replaceStrategy == RR) cout << "Using Random Replacement Strategy." << endl;
	else if (replaceStrategy == LRUR)
		cout << "Using Least Recently Used Replacement Strategy." << endl;
	else if (replaceStrategy == LFUR)
		cout << "Using Least Frequently Used Replacement Strategy." << endl;
	else if (replaceStrategy == FIFOR)
		cout << "Using FIFO Replacement Strategy." << endl;

	cout << GS << " lines in a set." << endl;
	Cache::printInfo();
}

inline bool GAMCache::isInCache(Address d_addr, int4_t & index)
{
	int baseIndex = d_addr.set * GS;
	for (int i = baseIndex; i < baseIndex + GS; i++) {
		if (memory[i].valid && memory[i].tag == d_addr.tag) {
			index = i;
			return true;
		}
	}
	return false;
}

inline void GAMCache::decode(address_t addr, Address & d_addr)
{
	address_t offsetMask = (1 << OBN) - 1;
	address_t offset = addr & offsetMask;
	address_t setMask = (1 << SBN) - 1;
	address_t set = (addr >> OBN) & setMask;
	address_t tag = addr >> (OBN + SBN);
	d_addr.offset = offset;
	d_addr.set = set;
	d_addr.tag = tag;
}

inline void GAMCache::handleReadMiss(const Address & d_addr)
{
	if (replaceStrategy == RR) RReplace(d_addr, false);
	else if (replaceStrategy == LFUR) LFUReplace(d_addr, false);
	else if (replaceStrategy == LRUR) LRUReplace(d_addr, false);
	else FIFOReplace(d_addr, false);
}

inline void GAMCache::handleWriteMiss(const Address & d_addr)
{
	if (replaceStrategy == RR) RReplace(d_addr, true);
	else if (replaceStrategy == LFUR) LFUReplace(d_addr, true);
	else if (replaceStrategy == LRUR) LRUReplace(d_addr, true);
	else FIFOReplace(d_addr, true);
}

inline void GAMCache::RReplace(const Address & d_addr, bit_t w)
{
	int baseIndex = d_addr.set * GS;
	int replacedLine = baseIndex + rand() % GS;

	if (memory[replacedLine].valid &&
		memory[replacedLine].dirty &&
		writeStrategy == WBack) {
		if (w) WTT += MAL;
		else RTT += MAL;
	}

	memory[replacedLine].valid = true;
	memory[replacedLine].tag = d_addr.tag;
	if (w) memory[replacedLine].dirty = true;
	else memory[replacedLine].dirty = false;
}

inline void GAMCache::LFUReplace(const Address & d_addr, bit_t w)
{
	int baseIndex = d_addr.set * GS;
	int4_t minIndex = baseIndex; int4_t minFrequency = memory[baseIndex].count;
	for (int i = baseIndex + 1; i < baseIndex + GS; i++) {
		if (memory[i].count < minFrequency) {
			minIndex = i;
			minFrequency = memory[i].count;
		}
		if (w) WTT += CPUCT;
		else RTT += CPUCT;
	}

	if (memory[minIndex].valid &&
		memory[minIndex].dirty &&
		writeStrategy == WBack) {
		if (w) WTT += MAL;
		else RTT += MAL;
	}

	memory[minIndex].valid = true;
	memory[minIndex].tag = d_addr.tag;
	memory[minIndex].count = 0;
	if (w) memory[minIndex].dirty = true;
	else memory[minIndex].dirty = false;
}

inline void GAMCache::LRUReplace(const Address & d_addr, bit_t w)
{
	int baseIndex = d_addr.set * GS;
	int4_t oldestIndex = baseIndex; int4_t oldest = memory[baseIndex].count;
	for (int i = baseIndex + 1; i < baseIndex + GS; i++) {
		if (memory[i].count > oldest) {
			oldestIndex = i;
			oldest = memory[i].count;
		}
		if (w) WTT += CPUCT;
		else RTT += CPUCT;
	}

	if (memory[oldestIndex].valid &&
		memory[oldestIndex].dirty &&
		writeStrategy == WBack) {
		if (w) WTT += MAL;
		else RTT += MAL;
	}

	memory[oldestIndex].valid = true;
	memory[oldestIndex].tag = d_addr.tag;
	memory[oldestIndex].count = 0;
	if (w) memory[oldestIndex].dirty = true;
	else memory[oldestIndex].dirty = false;
}

inline void GAMCache::FIFOReplace(const Address & d_addr, bit_t w)
{
	lastLoadIndex[d_addr.set]  = (lastLoadIndex[d_addr.set] + 1) % GS;
	int index = d_addr.set * GS + lastLoadIndex[d_addr.set];
	if (w) WTT += CPUCT;
	else RTT += CPUCT;

	if (memory[index].valid &&
		memory[index].dirty &&
		writeStrategy == WBack) {
		if (w) WTT += MAL;
		else RTT += MAL;
	}

	memory[index].valid = true;
	memory[index].tag = d_addr.tag;
	memory[index].count = 0;
	if (w) memory[index].dirty = true;
	else memory[index].dirty = false;
}