#pragma once
#include "type.h"

struct Address
{
	address_t tag;
	address_t set;
	address_t offset;
};

struct Line
{
	Line() : valid{ false }, count{ 0 } { }
	bit_t valid;
	bit_t dirty;
	address_t tag;
	byte4_t data[64];
	count_t count;
};

class Cache
{
public:
	Cache(size_t s, size_t ls, size_t ws, time_t cal, time_t mal, time_t cpuct);
	~Cache();
	void read(address_t addr);
	void write(address_t addr);
	void printInfo();
	void caculateInfo();
	void clear();

protected:
	// virtual void access(address_t addr) = 0;
	virtual bool isInCache(Address d_addr, int4_t& index) = 0;
	virtual void decode(address_t addr, Address& d_addr) = 0;
	virtual void handleWriteMiss(const Address& d_addr) = 0;
	virtual void handleReadMiss(const Address& d_addr) = 0;

protected:
	size_t S;  // size of cache, KB
	size_t LS; // size of cache line, B
	size_t WS; // word size, B

	int4_t LN; // total number of lines in cache
	int4_t LWN; // total number of word in a line

	time_t CAL; // cache access latency, ns
	time_t MAL; // memory access latency, ns
	time_t CPUCT; // time of a CPU cycle, ns

	count_t HC; // hit count
	count_t MC; // miss count
	count_t TABC; // total access byte count

	int4_t RC; // read count
	int4_t WC; // write count

	ratio_t HR; // hit ratio
	ratio_t MR; // miss ratio

	time_t RTT; // total time of reading cache, ns
	time_t WTT; // total time of writing data into cache, ns
	float4_t RAT; // average time of reading cache once, ns
	float4_t WAT; // average time of writing data into cache once, ns
	float4_t AAT; // average access time
	float4_t TP; // throughput, access byte from cache per second, MB/s

	ratio_t SUR; // speed up ratio

	ratio_t SU; // space utilization

	Line* memory; // 1D array of cache line simulating the memory block in cache
};

