#pragma once
#include <iostream>
#include <stdlib.h>
#include "Cache.h"

using namespace std;

class DMCache : public Cache
{
public:
	DMCache(size_t s, size_t ls, size_t ws, time_t cal, time_t mal, time_t cpuct);
	void printInfo();

protected:
	bool isInCache(Address d_addr, int4_t& index);
	void decode(address_t addr, Address& d_addr);
	void handleReadMiss(const Address& d_addr);
	void handleWriteMiss(const Address& d_addr);

	int4_t OBN; // number of bit in address's offset field 
	int4_t SBN; // number of bit in address's set field

};

DMCache::DMCache(size_t s, size_t ls, size_t ws, 
	time_t cal, time_t mal, time_t cpuct)
	: Cache{ s, ls, ws, cal, mal, cpuct }
{
	OBN = 0;
	int4_t tmp = LS;
	while (tmp != 1) {
		OBN++;
		tmp /= 2;
	}

	SBN = 0;
	tmp = LN;
	while (tmp != 1) {
		SBN++;
		tmp /= 2;
	}
}

inline void DMCache::printInfo()
{
	cout << "--------------Directed Mapping Cache--------------" << endl;
	if (writeStrategy == WBack) cout << "Using Write Back Strategy." << endl;
	else cout << "Using Write Through Strategy." << endl;

	Cache::printInfo();
}

inline bool DMCache::isInCache(Address d_addr, int4_t & index)
{
	if (memory[d_addr.set].valid && memory[d_addr.set].tag == d_addr.tag) {
		index = d_addr.set;
		return true;
	}
	else return false;
}

inline void DMCache::decode(address_t addr, Address & d_addr)
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

inline void DMCache::handleReadMiss(const Address & d_addr)
{
	if (memory[d_addr.set].valid &&
		memory[d_addr.set].dirty &&
		writeStrategy == WBack)
		RTT += MAL;

	memory[d_addr.set].valid = true;
	memory[d_addr.set].tag = d_addr.tag;
}

inline void DMCache::handleWriteMiss(const Address & d_addr)
{
	if (memory[d_addr.set].valid &&
		memory[d_addr.set].dirty &&
		writeStrategy == WBack)
		WTT += MAL;

	memory[d_addr.set].valid = true;
	memory[d_addr.set].dirty = true;
	memory[d_addr.set].tag = d_addr.tag;
}