#pragma once
#include <iostream>
#include <stdlib.h>
#include "type.h"
#include "Cache.h"

using namespace std;

class FAMCache : public Cache
{
public:
	FAMCache(size_t s, size_t ls, size_t ws, time_t cal, time_t mal, time_t cpuct);
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

	int4_t OBN; // number of bit in offset 

	int4_t lastLoadIndex; // used in FIFO replacement strategy
};

FAMCache::FAMCache(size_t s, size_t ls, size_t ws, time_t cal, time_t mal, time_t cpuct)
	: Cache{ s, ls, ws, cal, mal, cpuct }
{
	OBN = 0;
	int4_t tmp = LS;
	while (tmp != 1) {
		OBN++;
		tmp /= 2;
	}
	lastLoadIndex = -1;
}

inline void FAMCache::read(address_t addr)
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
			for (int i = 0; i < LN; i++) {
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

inline void FAMCache::write(address_t addr)
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
			for (int i = 0; i < LN; i++) {
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

inline void FAMCache::printInfo()
{
	cout << "----------Fully Associated Mapping Cache----------" << endl;
	if (writeStrategy == WBack) cout << "Using Write Back Strategy." << endl;
	else cout << "Using Write Through Strategy." << endl;

	if (replaceStrategy == RR) cout << "Using Random Replacement Strategy." << endl;
	else if (replaceStrategy == LRUR)
		cout << "Using Least Recently Used Replacement Strategy." << endl;
	else if (replaceStrategy == LFUR)
		cout << "Using Least Frequently Used Replacement Strategy." << endl;
	else if (replaceStrategy == FIFOR)
		cout << "Using FIFO Replacement Strategy." << endl;
	Cache::printInfo();

}

bool FAMCache::isInCache(Address d_addr, int4_t& index)
{
	for (int i = 0; i < LN; i++) {
		if (memory[i].valid && memory[i].tag == d_addr.tag) {
			index = i; return true;
		}
	}
	return false;
}

void FAMCache::decode(address_t addr, Address & d_addr)
{
	address_t offsetMask = (1 << OBN) - 1;
	address_t offset = addr & offsetMask;
	address_t tag = addr >> OBN;
	d_addr.offset = offset;
	d_addr.tag = tag;
}

void FAMCache::handleReadMiss(const Address & d_addr)
{
	if (replaceStrategy == RR) RReplace(d_addr, false);
	else if (replaceStrategy == LFUR) LFUReplace(d_addr, false);
	else if (replaceStrategy == LRUR) LRUReplace(d_addr, false);
	else FIFOReplace(d_addr, false);
}

inline void FAMCache::handleWriteMiss(const Address & d_addr)
{
	if (replaceStrategy == RR) RReplace(d_addr, true);
	else if (replaceStrategy == LFUR) LFUReplace(d_addr, true);
	else if (replaceStrategy == LRUR) LRUReplace(d_addr, true);
	else FIFOReplace(d_addr, true);
}

inline void FAMCache::RReplace(const Address & d_addr, bit_t w)
{
	int replacedLine = rand() % LN;
	if (w) WTT += CPUCT;
	else RTT += CPUCT;

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

inline void FAMCache::LFUReplace(const Address & d_addr, bit_t w)
{
	int4_t minIndex = 0; int4_t minFrequency = memory[0].count;
	for (int i = 1; i < LN; i++) {
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

inline void FAMCache::LRUReplace(const Address & d_addr, bit_t w)
{
	int4_t oldestIndex = 0; int4_t oldest = memory[0].count;
	for (int i = 1; i < LN; i++) {
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

inline void FAMCache::FIFOReplace(const Address & d_addr, bit_t w)
{
	lastLoadIndex = (lastLoadIndex + 1) % LN;
	if (w) WTT += CPUCT;
	else RTT += CPUCT;

	if (memory[lastLoadIndex].valid &&
		memory[lastLoadIndex].dirty &&
		writeStrategy == WBack) {
		if (w) WTT += MAL;
		else RTT += MAL;
	}

	memory[lastLoadIndex].valid = true;
	memory[lastLoadIndex].tag = d_addr.tag;
	memory[lastLoadIndex].count = 0;
	if (w) memory[lastLoadIndex].dirty = true;
	else memory[lastLoadIndex].dirty = false;
}
