#include "interation.h"

#include "AntTweakBar/AntTweakBar.h"
#include "glm/glm.hpp"
using namespace glm;

int rwVar = 10;
int roVar = 10;
unsigned char cubeColor[] = { 255, 0, 0, 128 };
vec3 gPosition1(-1.5f, 0.0f, 0.0f);
vec3 gOrientation1;

void TW_CALL buttonCallBackFunc(void* clientData)
{
	roVar = rwVar;
	printf("Button call back function....\n");
}


void Interator::initialize()
{
	TwInit(TW_OPENGL_CORE, nullptr);
	TwWindowSize(1024, 768);
	bar = TwNewBar("NameOfMyTweakBar");

	TwAddVarRW(bar, "Pos X", TW_TYPE_FLOAT, &gPosition1.x, "step=0.1");
	TwAddVarRW(bar, "Pos Y", TW_TYPE_FLOAT, &gPosition1.y, "step=0.1");
	TwAddVarRW(bar, "Euler Z", TW_TYPE_FLOAT, &gOrientation1.z, "step=0.01");
	TwAddButton(bar, "Testbutton", buttonCallBackFunc, nullptr, "help='Button help'");
	TwAddVarRW(bar, "Read-Write variable", TW_TYPE_BOOL32, &rwVar, "help='RW var help'");
	TwAddVarRO(bar, "Read-only variable", TW_TYPE_BOOL32, &roVar, "help='RO var help'");
	TwAddVarRW(bar, "cubeColor", TW_TYPE_COLOR32, &cubeColor,
		" label='Cube color' alpha help='Color and transparency of the cube.' ");

}

void Interator::update()
{
	TwDraw();
}

void Interator::destroy()
{
	TwTerminate();
}