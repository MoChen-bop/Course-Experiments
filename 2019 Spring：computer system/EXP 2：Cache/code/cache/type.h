#pragma once
#define replaceStrategy 3 // 1 for random replacement
                          // 2 for least frequently used replacement
                          // 3 for least recently used replacement
                          // 4 for first in first out replacement

#define RR 1
#define LFUR 2
#define LRUR 3
#define FIFOR 4

#define writeStrategy 1 // 1 for write back
                        // 2 for write through
#define WBack 1
#define WThrough 2

#define bit_t bool
#define time_t int
#define size_t int
#define address_t unsigned int
#define count_t int
#define ratio_t double
#define float4_t float
#define int4_t int
#define byte4_t int