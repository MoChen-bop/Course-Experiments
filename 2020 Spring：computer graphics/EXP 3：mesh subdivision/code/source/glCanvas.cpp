#include "glCanvas.h"
#include "mesh.h"
#include "argparser.h"
#include "camera.h"


ArgParser* GLCanvas::args = NULL;
Mesh* GLCanvas::mesh = NULL;
Camera* GLCanvas::camera = NULL;

int GLCanvas::mouseButton = 0;
int GLCanvas::mouseX = 0;
int GLCanvas::mouseY = 0;

bool GLCanvas::controlPressed = false;
bool GLCanvas::shiftPressed = false;
bool GLCanvas::altPressed = false;


void GLCanvas::initialize(ArgParser* _args, Mesh* _mesh)
{
	args = _args;
	mesh = _mesh;
	Vec3f camera_position = Vec3f(0, 0, 5);
	Vec3f point_of_interest = Vec3f(0, 0, 0);
	Vec3f up = Vec3f(0, 1, 0);
	camera = new PerspectiveCamera(camera_position, point_of_interest, up, 20 * M_PI / 100);

	glutInitWindowSize(args->width, args->height);
	glutInitWindowPosition(100, 100);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutCreateWindow("OpenGL Viewer");

	HandleGLError("in glcanvas initialize");

#ifdef _WIN32
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
	}
#endif

	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);
	glShadeModel(GL_SMOOTH);
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
	GLfloat ambient[] = { 0.2, 0.2, 0.2, 1.0 };
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glCullFace(GL_BACK);
	glDisable(GL_CULL_FACE);

	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);

	HandleGLError("finished glcanvas initialize");

	mesh->initializeVBOs();

	HandleGLError("finished glcanvas initialize");

	glutMainLoop();
}

void GLCanvas::InitLight()
{
	GLfloat position[4] = { 30, 30, 100, 1 };
	GLfloat diffuse[4] = { 0.75, 0.75, 0.75, 1 };
	GLfloat specular[4] = { 0, 0, 0, 1 };
	GLfloat ambient[4] = { 0.2, 0.2, 0.2, 1.0 };

	GLfloat zero[4] = { 0, 0, 0, 0 };
	glLightfv(GL_LIGHT1, GL_POSITION, position);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT1, GL_SPECULAR, specular);
	glLightfv(GL_LIGHT1, GL_AMBIENT, zero);
	glEnable(GL_LIGHT1);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glEnable(GL_COLOR_MATERIAL);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);

	GLfloat spec_mat[4] = { 1, 1, 1, 1 };
	float glexponent = 30;
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, &glexponent);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, spec_mat);

	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	float back_color[] = { 0.0, 0.0, 1.0, 1 };
	glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, back_color);
	glEnable(GL_LIGHT1);
}

void GLCanvas::display(void)
{
	glClearColor(1, 1, 1, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	InitLight();
	camera->glPlaceCamera();

	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);

	mesh->drawVBOs();

	glutSwapBuffers();
}

void GLCanvas::reshape(int w, int h)
{
	args->width = w;
	args->height = h;

	glViewport(0, 0, (GLsizei)args->width, (GLsizei)args->height);

	camera->glInit(args->width, args->height);
}

void GLCanvas::mouse(int button, int state, int x, int y)
{
	mouseButton = button;
	mouseX = x;
	mouseY = y;

	shiftPressed = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;
	controlPressed = (glutGetModifiers() & GLUT_ACTIVE_CTRL) != 0;
	altPressed = (glutGetModifiers() & GLUT_ACTIVE_ALT) != 0;
}

void GLCanvas::motion(int x, int y)
{
	if (controlPressed || shiftPressed || altPressed) {
		camera->zoomCamera(mouseY - y);
	}
	else if (mouseButton == GLUT_LEFT_BUTTON) {
		camera->rotateCamera(0.005 * (mouseX - x), 0.005 * (mouseY - y));
	}
	else if (mouseButton == GLUT_MIDDLE_BUTTON) {
		camera->truckCamera(mouseX - x, y - mouseY);
	}
	else if (mouseButton == GLUT_RIGHT_BUTTON) {
		camera->dollyCamera(mouseY - y);
	}
	mouseX = x;
	mouseY = y;

	glutPostRedisplay();
}

void GLCanvas::keyboard(unsigned char key, int x, int y)
{
	switch(key) {
	case 'w': case 'W':
		args->wireframe = !args->wireframe;
		glutPostRedisplay();
		break;
	case 'g' : case 'G':
		args->gouraud = !args->gouraud;
		mesh->setupVBOs();
		glutPostRedisplay();
		break;
	case 's': case 'S':
		mesh->LoopSubdivision();
		mesh->setupVBOs();
		glutPostRedisplay();
		break;
	case 'b': case 'B':
		mesh->BufferflySubdevision();
		mesh->setupVBOs();
		glutPostRedisplay();
		break;
	case 't': case 'T':
		mesh->TestLoopSubdivision();
		mesh->setupVBOs();
		glutPostRedisplay();
		break;
	case 'y': case 'Y':
		mesh->TestBufferflySubdivision();
		mesh->setupVBOs();
		glutPostRedisplay();
		break;
	case 'd': case 'D':
		mesh->Simplefication((int)floor(0.9 * mesh->numTriangles()));
		mesh->setupVBOs();
		glutPostRedisplay();
		break;
	case 'q': case'Q':
		exit(0);
		break;
	default:
		std::cout << "UNKNOWN KEYBOARD INPUT '" << key << "'" << std::endl;
	}
}

int HandleGLError(const std::string& message)
{
	GLenum error;
	int i = 0;
	while ((error = glGetError()) != GL_NO_ERROR) {
		if (message != "") {
			std::cout << "[" << message << "]";
		}
		std::cout << "GL_ERROR(" << i << ")" << gluErrorString(error) << std::endl;
		i++;
	}
	if (i == 0) return 1;
	return 0;
}
