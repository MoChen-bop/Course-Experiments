#pragma once

#include "AntTweakBar/AntTweakBar.h"

class Interator
{
public:
	Interator() : bar(nullptr) {}

	void initialize();
	void update();
	void destroy();
private:
	TwBar* bar;

	//void TW_CALL buttonCallBackFunc(void *clientData);
};