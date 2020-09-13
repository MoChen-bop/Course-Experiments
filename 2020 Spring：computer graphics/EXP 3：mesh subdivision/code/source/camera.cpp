#include "camera.h"
#include "glCanvas.h"
#include "matrix.h"

Camera::Camera(const Vec3f& c, const Vec3f& poi, const Vec3f& u)
{
	camera_position = c;
	point_of_interest = poi;
	up = u;
	up.Normalize();
}

void Camera::glPlaceCamera(void)
{
	gluLookAt(camera_position.x(), camera_position.y(), camera_position.z(),
		point_of_interest.x(), point_of_interest.y(), point_of_interest.z(),
		up.x(), up.y(), up.z());
}

void Camera::dollyCamera(double dist)
{
	Vec3f diff = camera_position - point_of_interest;
	double d = diff.Length();
	diff.Normalize();
	d *= pow(1.003, dist);
	camera_position = point_of_interest + diff * d;
}

void Camera::truckCamera(double dx, double dy)
{
	Vec3f diff = camera_position - point_of_interest;
	double d = diff.Length();
	Vec3f translate = (d * 0.0007) * (getHorizontal() * dx + getScreenUp() * dy);
	camera_position += translate;
	point_of_interest += translate;
}

void Camera::rotateCamera(double rx, double ry)
{
	double tiltAngle = acos(up.Dot3(getDirection()));
	if (tiltAngle - ry > 3.13)
		ry = tiltAngle - 3.13;
	else if (tiltAngle - ry < 0.01)
		ry = tiltAngle - 0.01;
	Matrix rotMat;
	rotMat.setToIdentity();
	rotMat *= Matrix::MakeTranslation(point_of_interest);
	rotMat *= Matrix::MakeAxisRotation(up, rx);
	rotMat *= Matrix::MakeAxisRotation(getHorizontal(), ry);
	rotMat *= Matrix::MakeTranslation(-point_of_interest);
	rotMat.Transform(camera_position);
}

std::ostream& operator<<(std::ostream& ostr, const Camera& c)
{
	const Camera* cp = &c;
	if (dynamic_cast<const OrthographicCamera*>(cp)) {
		const OrthographicCamera* ocp = (const OrthographicCamera*)cp;
		ostr << *ocp;
	}
	else if (dynamic_cast<const PerspectiveCamera*> (cp)) {
		const PerspectiveCamera* pcp = (const PerspectiveCamera*)cp;
		ostr << *pcp;
	}
	return ostr;
}

std::ostream& operator<<(std::ostream& ostr, const OrthographicCamera& c)
{
	ostr << "OrthographicCamera {" << std::endl;
	ostr << "    camera_position   " << c.camera_position;
	ostr << "    point_of_interest " << c.point_of_interest;
	ostr << "    up                " << c.up;
	ostr << "}" << std::endl;
	return ostr;
}

std::istream& operator>>(std::istream& istr, OrthographicCamera& c)
{
	std::string token;
	istr >> token; assert(token == "{");
	istr >> token; assert(token == "camera_position");
	istr >> c.camera_position;
	istr >> token; assert(token == "point_of_interest");
	istr >> c.point_of_interest;
	istr >> token; assert(token == "up");
	istr >> c.up;
	istr >> token; assert(token == "size");
	istr >> c.size;
	istr >> token; assert(token == "}");
	return istr;
}

std::ostream& operator<<(std::ostream& ostr, const PerspectiveCamera& c)
{
	ostr << "PerspectiveCamera {" << std::endl;
	ostr << "    camera_position    " << c.camera_position;
	ostr << "    point_of_interest  " << c.point_of_interest;
	ostr << "    up                 " << c.up;
	ostr << "    angle              " << c.angle << std::endl;
	ostr << "}" << std::endl;
	return ostr;
}

std::istream& operator>>(std::istream& istr, PerspectiveCamera& c)
{
	std::string token;
	istr >> token; assert(token == "{");
	istr >> token; assert(token == "camera_position");
	istr >> c.camera_position;
	istr >> token; assert(token == "point_of_interest");
	istr >> c.point_of_interest;
	istr >> token; assert(token == "up");
	istr >> c.up;
	istr >> token; assert(token == "angle");
	istr >> c.angle;
	istr >> token; assert(token == "}");
	return istr;
}

OrthographicCamera::OrthographicCamera(const Vec3f& c, const Vec3f& poi, const Vec3f& u, double s)
	: Camera(c, poi, u)
{
	size = s;
}

void OrthographicCamera::glInit(int w, int h)
{
	width = w;
	height = h;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	double horiz = size / 2.0;
	double vert = size / 2.0;
	double aspect = double(w) / double(h);
	if (aspect > 1)
		vert /= aspect;
	else
		horiz *= aspect;
	double dist_to_poi = (point_of_interest - camera_position).Length();
	glOrtho(-horiz, horiz, -vert, vert, dist_to_poi * 0.1, dist_to_poi * 10.0);
}

void OrthographicCamera::zoomCamera(double factor)
{
	size *= pow(1.005, factor);
	glInit(width, height);
}

PerspectiveCamera::PerspectiveCamera(const Vec3f& c, const Vec3f& poi, const Vec3f& u, double a)
	: Camera(c, poi, u)
{
	angle = a;
}

void PerspectiveCamera::glInit(int w, int h)
{
	width = w;
	height = h;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	double aspect = double(width) / double(height);
	double asp_angle = double(width) / double(height);
	if (aspect > 1) asp_angle /= aspect;
	double dist_to_poi = (point_of_interest - camera_position).Length();
	gluPerspective(asp_angle, aspect, dist_to_poi * 0.1, dist_to_poi * 100.0);
}

void PerspectiveCamera::zoomCamera(double dist)
{
	angle *= pow(1.003, dist);
	glInit(width, height);
}
