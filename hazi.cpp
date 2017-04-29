//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Fodor Balazs
// Neptun : GU87AO
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }

	void SetUniform(unsigned shaderProg, char * name) {
		int loc = glGetUniformLocation(shaderProg, name);   	
		glUniformMatrix4fv(loc, 1, GL_TRUE, &m[0][0]);
	}
};


mat4 Translate(float tx, float ty, float tz) {
	return mat4(1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		tx, ty, tz, 1);
}
mat4 Rotate(float angle, float wx, float wy, float wz) {
	return mat4(1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);
}
mat4 Scale(float sx, float sy, float sz) {
	return mat4(sx, 0, 0, 0,
		0, sy, 0, 0,
		0, 0, sz, 0,
		0, 0, 0, 1);
}



// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	vec4 operator*(const float f) {
		vec4 v= *this;
		for (int i=0;i<4;i++)
			v.v[i]*=f;
		return v;
	}

	vec4& operator=(const vec4& vec) {
		if (this != &vec) {
			this->v[0] = vec.v[0];
			this->v[1] = vec.v[1];
			this->v[2] = vec.v[2];
			this->v[3] = vec.v[3];
		}
		return *this;
	}
	void operator+=(const vec4& vec) {
		this->v[0] += vec.v[0];
		this->v[1] += vec.v[1];
		this->v[2] += vec.v[2];
	}
	
	vec4 operator-(const vec4& vec) {
		vec4 v = *this;
		v.v[0] -= vec.v[0];
		v.v[1] -= vec.v[1];
		v.v[2] -= vec.v[2];
		return v;
	}

	vec4 operator+(const vec4& vec) {
		vec4 v = *this;
		v.v[0] += vec.v[0];
		v.v[1] += vec.v[1];
		v.v[2] += vec.v[2];
		return v;
	}
	float length() {
		return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}

	void norm() {
		float length = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
		v[0] *= 1 / length;
		v[1] *= 1 / length;
		v[2] *= 1 / length;
	}

	/*
	
	float x, y, z;

	vec3 inv() {
		float x0 = 0.0, y0 = 0.0, z0 = 0.0;
		if (x != 0.0) x0 = 1 / x;
		if (y != 0.0) y0 = 1 / y;
		if (z != 0.0) z0 = 1 / z;
		return vec3(x0,y0,z0);
	}

	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }

	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }

	vec3 operator+(const vec3& v) const {
		return vec3(x + v.x, y + v.y, z + v.z);
	}
	vec3 operator-(const vec3& v) const {
		return vec3(x - v.x, y - v.y, z - v.z);
	}
	vec3 operator*(const vec3& v) const {
		return vec3(x * v.x, y * v.y, z * v.z);
	}
	vec3 operator-() const {
		return vec3(-x, -y, -z);
	}
	vec3 normalize() const {
		return (*this) * (1 / (Length() + 0.000001f));
	}
	float Length() const { return sqrtf(x * x + y * y + z * z); }

	operator float*() { return &x; }*/


};

float dot(const vec4& v1, const vec4& v2) {
	return v1.v[0] * v2.v[0] + v1.v[1] * v2.v[1] + v1.v[2] * v2.v[2];
}

vec4 cross(const vec4& v1, const vec4& v2) {
	return vec4(v1.v[1] * v2.v[2] - v1.v[2] * v2.v[1], v1.v[2] * v2.v[0] - v1.v[0] * v2.v[2], v1.v[0] * v2.v[1] - v1.v[1] * v2.v[0]);
}

//VEC3
struct vec3 {
	float x, y, z;

	vec3 inv() {
		float x0 = 0.0, y0 = 0.0, z0 = 0.0;
		if (x != 0.0) x0 = 1 / x;
		if (y != 0.0) y0 = 1 / y;
		if (z != 0.0) z0 = 1 / z;
		return vec3(x0, y0, z0);
	}

	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }

	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }

	vec3 operator+(const vec3& v) const {
		return vec3(x + v.x, y + v.y, z + v.z);
	}
	vec3 operator-(const vec3& v) const {
		return vec3(x - v.x, y - v.y, z - v.z);
	}
	vec3 operator*(const vec3& v) const {
		return vec3(x * v.x, y * v.y, z * v.z);
	}
	vec3 operator-() const {
		return vec3(-x, -y, -z);
	}
	vec3 normalize() const {
		return (*this) * (1 / (Length() + 0.000001f));
	}
	float Length() const { return sqrtf(x * x + y * y + z * z); }

	operator float*() { return &x; }
};

float dot(const vec3& v1, const vec3& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

//----
struct Material {
	vec4 kd, ks, ka;// diffuse, specular, ambient ref
	vec4 La, Le;    // ambient and point source rad
	float shine;    // shininess for specular ref

	Material(vec4 kd, vec4 ks, vec4 ka, vec4 La, vec4 Le, float shine) {
		this->kd = kd;
		this->ks = ks;
		this->ka = ka;
		this->La = La;
		this->Le = Le;
		this->shine = shine;
	}
};
//
struct Geometry {
	unsigned int vao, nVtx;

	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, nVtx);
	}
};

struct VertexData {
	/*
	vec3 kent kell hasznalni
	*/
	vec3 position, normal;
	float u, v;
};

struct ParamSurface : Geometry {
	virtual VertexData GenVertexData(float u, float v) = 0;
	void Create(int N, int M);
};

//NxM kocskabol allo negyzetracsos halon kepez 0-1 intervallumon pontokat
void ParamSurface::Create(int N, int M) {
	nVtx = N * M * 6;
	unsigned int vbo;
	glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);

	VertexData *vtxData = new VertexData[nVtx], *pVtx = vtxData;
	for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) {
		*pVtx++ = GenVertexData((float)i / N, (float)j / M);
		*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
		*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
		*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
		*pVtx++ = GenVertexData((float)(i + 1) / N, (float)(j + 1) / M);
		*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
	}
	glBufferData(GL_ARRAY_BUFFER,
		nVtx * sizeof(VertexData), vtxData, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);  // AttribArray 0 = POSITION
	glEnableVertexAttribArray(1);  // AttribArray 1 = NORMAL
	glEnableVertexAttribArray(2);  // AttribArray 2 = UV
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
		sizeof(VertexData), (void*)offsetof(VertexData, position));
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
		sizeof(VertexData), (void*)offsetof(VertexData, normal));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
		sizeof(VertexData), (void*)offsetof(VertexData, u));
}

struct Light {
	vec3 La, Le;
	vec3 wLightPos;
	Light() {
		La = vec3(10000, 10000, 10000);
		Le = vec3(10000, 10000, 10000);
		wLightPos = vec3(0,100,0);

	}

};

struct RenderState {
	mat4 M, V, P, Minv;
	Material* material;
	//Texture texture;
	Light light;
	vec3 wEye;
};

//3D camera
class Camera {
public:
	vec3  wEye, wLookat, wVup;
	float fov, asp, fp, bp; //fp / fb nem lehet kicsi !!!

	Camera() {
		wEye=vec3(0,0,-100);
		wLookat = vec3(0, 0, -20);
		wLookat = vec3(0, 0, -20);
		fov = 200; // vizszintes szélesség
		asp = 200;// függőleges magasság
		fp = 20;//
		bp = 200; //

	}

	mat4 V() { // view matrix
		vec3 w = (wEye - wLookat).normalize();
		vec3 u = cross(wVup, w).normalize();
		vec3 v = cross(w, u);
		return Translate(-wEye.x, -wEye.y, -wEye.z) *
			mat4(u.x, v.x, w.x, 0.0f,
				u.y, v.y, w.y, 0.0f,
				u.z, v.z, w.z, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f);
	}
	mat4 P() { // projection matrix
		float sy = 1 / tan(fov / 2);
		return mat4(sy / asp, 0.0f, 0.0f, 0.0f,
			0.0f, sy, 0.0f, 0.0f,
			0.0f, 0.0f, -(fp + bp) / (bp - fp), -1.0f,
			0.0f, 0.0f, -2 * fp*bp / (bp - fp), 0.0f);
	}
};

struct Shader {
	unsigned int shaderProg;

	void Create(const char * vsSrc,
		const char * fsSrc, const char * fsOuputName) {
		GLenum err = glewInit();
		unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vs, 1, &vsSrc, NULL); glCompileShader(vs);
		unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs, 1, &fsSrc, NULL); glCompileShader(fs);
		shaderProg = glCreateProgram();
		glAttachShader(shaderProg, vs);
		glAttachShader(shaderProg, fs);

		glBindFragDataLocation(shaderProg, 0, fsOuputName);
		glLinkProgram(shaderProg);
	}
	virtual
		void Bind(RenderState& state) { glUseProgram(shaderProg); }
};

class PhongShader : public Shader {
	const char * vsSrc = R"(
	uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
	uniform vec4  kd, ks, ka;   // diffuse, specular, ambient ref
	uniform vec4  La, Le;       // ambient and point sources
	uniform vec4  wLiPos;       // pos of light source in world
	uniform vec3  wEye;         // pos of eye in world
	uniform float shine;		 // shininess for specular ref

	layout(location = 0) in vec3 vtxPos;  // pos in modeling space
	layout(location = 1) in vec3 vtxNorm; // normal in modeling space
	out vec4 color;             // computed vertex color

	void main() {
	   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC

	   vec4 wPos = vec4(vtxPos, 1) * M;
	   vec3 L = normalize( wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w);
	   vec3 V = normalize(wEye * wPos.w - wPos.xyz);
	   vec4 wNormal = Minv * vec4(vtxNorm, 0);
	   vec3 N = normalize(wNormal.xyz);
	   vec3 H = normalize(L + V);
	   float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
	   color = ka * La + (kd * cost + ks * pow(cosd, shine)) * Le;
	}

)";

	const char * fsSrc = R"(
	in vec4 color;          // interpolated color of vertex shader
	out vec4 fragmentColor; // output goes to frame buffer

	void main() {
	   fragmentColor = color; 
	}

)";
public:
	PhongShader() {
		Create(vsSrc, fsSrc, "fragmentColor");
	}

	void Bind(RenderState& state) {
		glUseProgram(shaderProg);
		mat4 MVP = state.M * state.V * state.P;
		MVP.SetUniform(shaderProg, "MVP");
	}
};


class Object{
public:
	Shader *   shader;
	Material * material;
	//Texture *  texture;
	Geometry * geometry;
	vec3 scale, pos, rotAxis;
	float rotAngle;

	virtual void Draw(RenderState state) {
		state.M = Scale(scale.x, scale.y, scale.z) *
			Rotate(rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
			Translate(pos.x, pos.y, pos.z);
		state.Minv = Translate(-pos.x, -pos.y, -pos.z) *
			Rotate(-rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
			Scale(1 / scale.x, 1 / scale.y, 1 / scale.z);
		state.material = material; 
		//state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}
	virtual void Animate(float dt) {}
};



/*
class LagrangeCurve {
	GLuint vao, vbo[2];
	float points[2000]; 
	float colors[3000];
	int nVertices=0; 
	std::vector<vec4> vertexData;
	std::vector<vec4> vertexVectors;
	std::vector<float> ts;
	float db=100;
	float width = 1.0f;
	float scale = 1.0f;
	mat4 trans = { 1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		-(float)width / 2,-(float)width / 2,0,1 };
	mat4 scalet = { scale,0,0,0,
		0,scale,0,0,
		0,0,scale,0,
		0,0,0,1 };
	mat4 transinv = { 1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		(float)width / 2,(float)width / 2,0,1 };
	mat4 scaletinv = { 1/scale,0,0,0,
		0,1/scale,0,0,
		0,0,1/scale,0,
		0,0,0,1 };

	float L(int i, float t) {
		float Li = 1.0f;
		for (size_t j = 0; j < vertexData.size(); j++)
			if (j != i) Li *= (t - ts[j]) / (ts[i] - ts[j]);
		return Li;
	}

	float dL(int i,float t) {
		float a = 0.0f;
		for (size_t j = 0; j < vertexData.size(); j++){
			if (j != i) {
				a += 1 / (t - ts[j]);
			}
		}
		return a*L(i, t);
	}
public:
	vec4 r(float t) {
		vec4 rr(0, 0, 0);
		for (size_t i = 0; i < vertexData.size(); i++) {
			rr += (vertexData[i] * L(i, t));
		}
		return rr;
	}
	vec4 dr(float t) {
		vec4 rr(0, 0, 0);
		for (size_t i = 0; i < vertexData.size(); i++) {
			rr += (vertexData[i] * dL(i, t));
		}

		return rr;
	}


	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(2, &vbo[0]); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL); // attribute array, components/attribute, component type, normalize?, stride, offset
		
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,0,NULL);

		for (int i = 0; i < 3000; i++) {
			colors[i] = 1.0f;
		}

		glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);
	}

	void AddPoint(float cX, float cY, float ti) {

		if (ti < 0.0) ti = (float)vertexData.size();


		vec4 wVertex = vec4(cX, cY, 0, 1);//* camera.Pinv() * camera.Vinv()*scaletinv*transinv;//0-1 es
		//wVertex.v[2] = beziersurf.Z(wVertex.v[0], wVertex.v[1]);
		vertexData.push_back(wVertex);
		ts.push_back(ti);
		if (vertexData.size() < 5) return;

		//add a vectors to vertices
		vertexVectors.push_back(vec4(1, 0, 0, 0));
		for (int i = 1; i < vertexData.size() - 1; i++) {
			vertexVectors.push_back(
				((vertexData[i + 1] - vertexData[i]) * (1 / (ts[i + 1] - ts[i])) +
				(vertexData[i] - vertexData[i - 1]) * (1 / (ts[i] - ts[i - 1])))* 0.5
			);
		}
		vertexVectors.push_back(vec4(1, 0, 0, 0));
		//
		//DRAW THE LINE
		int j = 0;
		for (int i = 0; i < vertexData.size() - 1; i++) {
			vec4 p0 = vertexData[i], p1 = vertexData[i+1], v0 = vertexVectors[i],
				v1 = vertexVectors[i+1];
			//
			vec4 a0 = p0;
			vec4 a1 = v0;
			vec4 a2 = (p1 - p0)*(3.0f / ((ts[1] - ts[0])*(ts[1] - ts[0])))
				- (v1 + v0*2.0f)*(1 /
				(ts[1] - ts[0]));
			vec4 a3 = (p0 - p1)*(2 /
				pow(ts[1] - ts[0], 3)) + (v1 + v0)*(1 /
				((ts[1] - ts[0])*(ts[1] - ts[0])));

			
			for (float t = ts[0]; t < ts[1]; t += 0.1) {
				vec4 v = a3*pow(t - ts[0], 3) + a2*pow(t - ts[0], 2) + a1*(t - ts[0]) + a0;
				//v.v[2] = beziersurf.Z(v.v[0], v.v[1]);

				points[j++] = v.v[0];
				points[j++] = v.v[1];
				
			}
		}
		nVertices = j / 2;

		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_DYNAMIC_DRAW);
		
	}

	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = trans*scalet*camera.V() * camera.P();

			//int location = glGetUniformLocation(shaderProgram, "MVP");
			//if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			//else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
		}
	}

	float getEndT() { 
		return ts[ts.size() - 1];  
	}

};
*/
/*
class Snake:public ParamSurface {


	void Draw() {
		mat4 M = Scale(scale.x, scale.y, scale.z) *
			Rotate(rotAng, rotAxis.x, rotAxis.y, rotAxis.z) *
			Translate(pos.x, pos.y, pos.z);
		mat4 Minv = Translate(-pos.x, -pos.y, -pos.z) *
			Rotate(-rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
			Scale(1 / scale.x, 1 / scale.y, 1 / scale.z);
		mat4 MVP = M * camera.V() * camera.P();

		M.SetUniform(shaderProg, "M");
		Minv.SetUniform(shaderProg, "Minv");
		MVP.SetUniform(shaderProg, "MVP");

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, nVtx);
	}

};
*/
class Sphere : public ParamSurface {
	vec3 center;
	float radius;
public:
	Sphere(vec3 c, float r) : center(c), radius(r) {
		Create(16, 8); // tessellation level
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(cos(u * 2 * M_PI) * sin(v*M_PI),
			sin(u * 2 * M_PI) * sin(v*M_PI),
			cos(v*M_PI));
		vd.position = vd.normal * radius + center;
		vd.u = u; vd.v = v;
		return vd;
	}
};

class SargaGomb : public Object {
public :
	SargaGomb() {
		shader = new PhongShader();
		material = new Material(vec4(1,0,0,1), vec4(1,1,1,1), vec4(1,0,0,1), 
			vec4(1,1,1,1), vec4(1,1,1,1), 2);
		geometry = new Sphere(vec3(), 50);
	}
	void Draw(RenderState state) {
		int location = glGetUniformLocation(shader->shaderProg, "kd");
		if (location >= 0) glUniform4fv(location, 1, material->kd.v);
		location = glGetUniformLocation(shader->shaderProg, "ka");
		if (location >= 0) glUniform4fv(location, 1, material->ka.v);
	     location = glGetUniformLocation(shader->shaderProg, "ks");
		if (location >= 0) glUniform4fv(location, 1, material->ks.v);
		location = glGetUniformLocation(shader->shaderProg, "shine");
		if (location >= 0) glUniform1f(location, material->shine);
		location = glGetUniformLocation(shader->shaderProg, "La");
		if (location >= 0) glUniform4fv(location, 1, material->La.v);
		location = glGetUniformLocation(shader->shaderProg, "Le");
		if (location >= 0) glUniform4fv(location, 1, material->Le.v);
		location = glGetUniformLocation(shader->shaderProg, "wLiPos");
		if (location >= 0) glUniform4fv(location, 1, vec4(state.light.wLightPos.x, state.light.wLightPos.y, state.light.wLightPos.z, 1).v);
		location = glGetUniformLocation(shader->shaderProg, "wEye");
		if (location >= 0) glUniform4fv(location, 1, state.wEye);
		Object::Draw(state);
	}

};

class Scene {
	Camera camera = Camera();
	std::vector<Object *> objects;
	Light light;
public:
	void Create() {
		objects.push_back(new SargaGomb());
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.light = light;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Animate(float dt) {
		for (Object * obj : objects) obj->Animate(dt);
	}
};
Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.Create();
	// Create objects by setting up their vertex data on the GPU
	/*
	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

	// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
	*/
}

void onExit() {
	//glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	//Ide jon a rajzolas

	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == ' ') {
		long time = glutGet(GLUT_ELAPSED_TIME); // elapsed ti
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP

	}
} 

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec

	glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif
	glEnable(GL_DEPTH_TEST); // z-buffer is on
	glDisable(GL_CULL_FACE); // backface culling is off


	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

