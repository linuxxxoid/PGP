#include <stdlib.h>
#include <stdio.h> 
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <string.h>


using namespace std::chrono;


#define CSC(call) do { \
	cudaError_t res = call;	\
	if (res != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
		fflush(stderr); \
		exit(0); \
	} \
} while (0) \



typedef unsigned char uchar;

struct vec3
{
	double x;
	double y;
	double z;
};


struct Triangle
{
	vec3 a;
	vec3 b;
	vec3 c;
	uchar4 color;
};


__device__ __host__
double dot(vec3 a, vec3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}


__device__ __host__
vec3 mulc(vec3 a, double c)
{
	return { c * a.x, c * a.y, c * a.z };
}


__device__ __host__
vec3 prod(vec3 a, vec3 b)
{
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}


__device__ __host__
vec3 norm(vec3 v)
{
	double l = sqrt(dot(v, v));
	return { v.x / l, v.y / l, v.z / l };
}


__device__ __host__
double len(vec3 v)
{
	return sqrt(dot(v, v));
}


__device__ __host__
vec3 diff(vec3 a, vec3 b)
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}


__device__ __host__
vec3 add(vec3 a, vec3 b)
{
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}


__device__ __host__
vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v)
{
	return  {
				a.x * v.x + b.x * v.y + c.x * v.z,
				a.y * v.x + b.y * v.y + c.y * v.z,
				a.z * v.x + b.z * v.y + c.z * v.z
			};
}


void print(vec3 v)
{
	printf("%e %e %e\n", v.x, v.y, v.z);
}


__host__ __device__
double dmin(double x, double y)
{
	if (x > y) { return y; }
	return x;
}


void BuildStage(Triangle* t, double r1, vec3 o1, uchar4 c1, 
				double r2, vec3 o2, uchar4 c2, 
				double r3, vec3 o3, uchar4 c3, 
				vec3* fv, uchar4 fc)
{
	int st = 0;
	double p = (1 + sqrt(5)) / 2;

	vec3 icosVertexes[] = {
		add(o1, norm(vec3{ 0, -1,  p})),
		add(o1, norm(vec3{ 0,  1,  p})),
		add(o1, norm(vec3{-p,  0,  1})),
		add(o1, norm(vec3{ p,  0,  1})),
		add(o1, norm(vec3{-1,  p,  0})),
		add(o1, norm(vec3{ 1,  p,  0})),
		add(o1, norm(vec3{ 1, -p,  0})),
		add(o1, norm(vec3{-1, -p,  0})),
		add(o1, norm(vec3{-p,  0, -1})),
		add(o1, norm(vec3{ p,  0, -1})),
		add(o1, norm(vec3{ 0, -1, -p})),
		add(o1, norm(vec3{ 0,  1, -p}))
	};

	t[st++] = Triangle{ icosVertexes[0],  icosVertexes[1],  icosVertexes[2] , c1 };
	t[st++] = Triangle{ icosVertexes[1],  icosVertexes[0],  icosVertexes[3] , c1 };
	t[st++] = Triangle{ icosVertexes[0],  icosVertexes[2],  icosVertexes[7] , c1 };
	t[st++] = Triangle{ icosVertexes[2],  icosVertexes[1],  icosVertexes[4] , c1 };
	t[st++] = Triangle{ icosVertexes[4],  icosVertexes[1],  icosVertexes[5] , c1 };
	t[st++] = Triangle{ icosVertexes[6],  icosVertexes[0],  icosVertexes[7] , c1 };
	t[st++] = Triangle{ icosVertexes[3],  icosVertexes[0],  icosVertexes[6] , c1 };
	t[st++] = Triangle{ icosVertexes[1],  icosVertexes[3],  icosVertexes[5] , c1 };
	t[st++] = Triangle{ icosVertexes[4],  icosVertexes[5],  icosVertexes[11], c1 };
	t[st++] = Triangle{ icosVertexes[6],  icosVertexes[7],  icosVertexes[10], c1 };
	t[st++] = Triangle{ icosVertexes[3],  icosVertexes[6],  icosVertexes[9] , c1 };
	t[st++] = Triangle{ icosVertexes[5],  icosVertexes[3],  icosVertexes[9] , c1 };
	t[st++] = Triangle{ icosVertexes[7],  icosVertexes[2],  icosVertexes[8] , c1 };
	t[st++] = Triangle{ icosVertexes[2],  icosVertexes[4],  icosVertexes[8] , c1 };
	t[st++] = Triangle{ icosVertexes[9],  icosVertexes[10], icosVertexes[11], c1 };
	t[st++] = Triangle{ icosVertexes[10], icosVertexes[8],  icosVertexes[11], c1 };
	t[st++] = Triangle{ icosVertexes[5],  icosVertexes[9],  icosVertexes[11], c1 };
	t[st++] = Triangle{ icosVertexes[9],  icosVertexes[6],  icosVertexes[10], c1 };
	t[st++] = Triangle{ icosVertexes[7],  icosVertexes[8],  icosVertexes[10], c1 };
	t[st++] = Triangle{ icosVertexes[8],  icosVertexes[4],  icosVertexes[11], c1 };

	vec3 hexVertexes[] = {
		add(o2, norm(vec3{-1, -1,  -1})),
		add(o2, norm(vec3{-1,  1,  -1})),
		add(o2, norm(vec3{ 1, -1,  -1})),
		add(o2, norm(vec3{ 1,  1,  -1})),
		add(o2, norm(vec3{-1, -1,   1})),
		add(o2, norm(vec3{-1,  1,   1})),
		add(o2, norm(vec3{ 1, -1,   1})),
		add(o2, norm(vec3{ 1,  1,   1})),
	};

	t[st++] = Triangle{ hexVertexes[5],  hexVertexes[4],  hexVertexes[7], c2 };
	t[st++] = Triangle{ hexVertexes[4],  hexVertexes[6],  hexVertexes[7], c2 };
	t[st++] = Triangle{ hexVertexes[0],  hexVertexes[1],  hexVertexes[3], c2 };
	t[st++] = Triangle{ hexVertexes[2],  hexVertexes[0],  hexVertexes[3], c2 };
	t[st++] = Triangle{ hexVertexes[3],  hexVertexes[1],  hexVertexes[5], c2 };
	t[st++] = Triangle{ hexVertexes[3],  hexVertexes[5],  hexVertexes[7], c2 };
	t[st++] = Triangle{ hexVertexes[0],  hexVertexes[2],  hexVertexes[4], c2 };
	t[st++] = Triangle{ hexVertexes[4],  hexVertexes[2],  hexVertexes[6], c2 };
	t[st++] = Triangle{ hexVertexes[1],  hexVertexes[0],  hexVertexes[5], c2 };
	t[st++] = Triangle{ hexVertexes[0],  hexVertexes[4],  hexVertexes[5], c2 };
	t[st++] = Triangle{ hexVertexes[2],  hexVertexes[3],  hexVertexes[7], c2 };
	t[st++] = Triangle{ hexVertexes[6],  hexVertexes[2],  hexVertexes[7], c2 };


	vec3 octVertexes[] = {
		add(o3, norm(vec3{ 0,  0, -1})),
		add(o3, norm(vec3{ 0,  0,  1})),
		add(o3, norm(vec3{-1,  0,  0})),
		add(o3, norm(vec3{ 1,  0,  0})),
		add(o3, norm(vec3{ 0, -1,  0})),
		add(o3, norm(vec3{ 0,  1,  0})),
	};

	t[st++] = Triangle{ octVertexes[0],  octVertexes[4],  octVertexes[2], c3 };
	t[st++] = Triangle{ octVertexes[0],  octVertexes[2],  octVertexes[5], c3 };
	t[st++] = Triangle{ octVertexes[0],  octVertexes[5],  octVertexes[3], c3 };
	t[st++] = Triangle{ octVertexes[0],  octVertexes[3],  octVertexes[4], c3 };
	t[st++] = Triangle{ octVertexes[1],  octVertexes[2],  octVertexes[4], c3 };
	t[st++] = Triangle{ octVertexes[1],  octVertexes[5],  octVertexes[2], c3 };
	t[st++] = Triangle{ octVertexes[1],  octVertexes[3],  octVertexes[5], c3 };
	t[st++] = Triangle{ octVertexes[1],  octVertexes[4],  octVertexes[3], c3 };

	t[st++] = Triangle{ fv[0],  fv[2],  fv[1], fc };
	t[st++] = Triangle{ fv[1],  fv[2],  fv[3], fc };
}


__host__ __device__
void RayTracing(Triangle* triangles, vec3 pos, vec3 dir, int* i, double* t)
{
	int k, k_min = -1;
	double ts_min = 0;
	for (k = 0; k < 42; ++k) 
	{
		vec3 e1 = diff(triangles[k].b, triangles[k].a);
		vec3 e2 = diff(triangles[k].c, triangles[k].a);
		vec3 p = prod(dir, e2);
		double div = dot(p, e1);
		if (fabs(div) < 1e-10)
			continue; 
		vec3 t = diff(pos, triangles[k].a);
		double u = dot(p, t) / div;
		if (u < 0.0 || u > 1.0)
			continue;
		vec3 q = prod(t, e1);
		double v = dot(q, dir) / div;
		if (v < 0.0 || v + u > 1.0)
			continue;
		double ts = dot(q, e2) / div;
		if (ts < 0.0)
			continue;
		if (k_min == -1 || ts < ts_min) 
		{
			k_min = k;
			ts_min = ts;
		}
	}
	*i = k_min;
	*t = ts_min;
}


void Render(Triangle* triangles, vec3 pc, vec3 pv,
			int w, int h, double angle, uchar4* data,
			vec3 lightPosition, uchar4 lightColor)
{
	double pi = acos(-1.0);
	int i, j;
	double dw = 2.0 / (w - 1.0);
	double dh = 2.0 / (h - 1.0);
	double z = 1.0 / tan(angle * pi / 360.0);

	vec3 bz = norm(diff(pv, pc));
	vec3 bx = norm(prod(bz, { 0.0, 0.0, 1.0 }));
	vec3 by = norm(prod(bx, bz));

	int kmin;
	double tmin;

	for (i = 0; i < w; i++)
		for (j = 0; j < h; j++)
		{
			vec3 v = { -1.0 + dw * i, (-1.0 + dh * j) * h / w, z };

			vec3 dir = norm(mult(bx, by, bz, v));

			RayTracing(triangles, pc, dir, &kmin, &tmin);
			if (kmin != -1)
			{
				double rr = (double)triangles[kmin].color.x / 255.0;
				double gg = (double)triangles[kmin].color.y / 255.0;
				double bb = (double)triangles[kmin].color.z / 255.0;
				double ri = 0.2, gi = 0.2, bi = 0.2;

				vec3 p = add(pc, mulc(dir, tmin));
				vec3 l = diff(lightPosition, p);
				vec3 n = prod(diff(triangles[kmin].b, triangles[kmin].a), 
					          diff(triangles[kmin].c, triangles[kmin].a));
				double dot_nl = dot(n, l);

				if (dot_nl > 0)
				{
					ri += (lightColor.x / 255.0) * dot_nl / (len(n) * len(l));
					gi += (lightColor.y / 255.0) * dot_nl / (len(n) * len(l));
					bi += (lightColor.z / 255.0) * dot_nl / (len(n) * len(l));
				}
				data[(h - 1 - j) * w + i].x = (uchar)(255 * dmin(1.0, ri * rr));
				data[(h - 1 - j) * w + i].y = (uchar)(255 * dmin(1.0, gi * gg));
				data[(h - 1 - j) * w + i].z = (uchar)(255 * dmin(1.0, bi * bb));
			}
			else
			{
				data[(h - 1 - j) * w + i] = uchar4{ 0, 0, 0, 0 };
			}
		}
}


__global__
void DeviceRender(Triangle* triangles, vec3 pc, vec3 pv,
				  int w, int h, double angle, uchar4* data,
				  vec3 lightPosition, uchar4 lightColor)
{
	double pi = acos(-1.0);
	int i, j;
	double dw = 2.0 / (w - 1.0);
	double dh = 2.0 / (h - 1.0); 
	double z = 1.0 / tan(angle * pi / 360.0); 
	
	vec3 bz = norm(diff(pv, pc)); 
	vec3 bx = norm(prod(bz, { 0.0, 0.0, 1.0 })); 
	vec3 by = norm(prod(bx, bz)); 

	int kmin;
	double tmin;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ofs = blockDim.x * gridDim.x;

	while (tid < w * h)
	{
		i = tid % w;
		j = tid / w;

		tid += ofs;

		vec3 v = { -1.0 + dw * i, (-1.0 + dh * j) * h / w, z };

		vec3 dir = norm(mult(bx, by, bz, v));

		RayTracing(triangles, pc, dir, &kmin, &tmin);
		if (kmin != -1)
		{
			double rr = (double)triangles[kmin].color.x / 255.0;
			double gg = (double)triangles[kmin].color.y / 255.0;
			double bb = (double)triangles[kmin].color.z / 255.0;
			double ri = 0.2, gi = 0.2, bi = 0.2;

			vec3 p = add(pc, mulc(dir, tmin));
			vec3 l = diff(lightPosition, p);
			vec3 n = prod(diff(triangles[kmin].b, triangles[kmin].a),
					 	  diff(triangles[kmin].c, triangles[kmin].a));
			double dot_nl = dot(n, l);

			if (dot_nl > 0)
			{
				ri += (lightColor.x / 255.0) * dot_nl / (len(n) * len(l));
				gi += (lightColor.y / 255.0) * dot_nl / (len(n) * len(l));
				bi += (lightColor.z / 255.0) * dot_nl / (len(n) * len(l));
			}
			data[(h - 1 - j) * w + i].x = (uchar)(255 * dmin(1.0, ri * rr));
			data[(h - 1 - j) * w + i].y = (uchar)(255 * dmin(1.0, gi * gg));
			data[(h - 1 - j) * w + i].z = (uchar)(255 * dmin(1.0, bi * bb));
		}
		else
		{
			data[(h - 1 - j) * w + i] = uchar4{ 0, 0, 0, 0 };
		}
	}
}


vec3 CoordCameraFromTime(double r0c, double z0c, double p0c,
						 double arc, double azc, 
						 double wrc, double wzc, double wpc,
						 double prc, double pzc, double t)
{
	double r = r0c + arc * sin(wrc * t + prc);
	double z = z0c + azc * sin(wzc * t + pzc);
	double phi = p0c + wpc * t;
	return vec3{ r * cos(phi), r * sin(phi), z };
};


vec3 CoordViewPointFromTime(double r0n, double z0n, double p0n,
							double arn, double azn,
							double wrn, double wzn, double wpn,
							double prn, double pzn, double t)
{
	double r = r0n + arn * sin(wrn * t + prn);
	double z = z0n + azn * sin(wzn * t + pzn);
	double phi = p0n + wpn * t;
	return vec3{ r * cos(phi), r * sin(phi), z };
};


int main(int argc, char* argv[])
{
	int deviceSelection = 0;
	if (argc >= 3)
	{
		printf("argc error\n");
		return -1;
	}
	if (argc == 1)
	{
		deviceSelection = 1;
	}
	else if (strcmp(argv[1], "--default") == 0)
	{
		printf("400                                             \n");
		printf("img_%% d.data                                   \n");
		printf("1240 960 100                                    \n");
		printf("7.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0         \n");
		printf("2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0         \n");
		printf("-2 -2  0  2 200 0 0                             \n");
		printf("-2  2  0  2 0 255 0                             \n");
		printf("2  0  0  2 0 0 255                              \n");
		printf("-4 -4 -1 -4 4 -1 4 -4 -1 4 4 -1 102 62 0        \n");
		return 0;
	}
	else if (strcmp(argv[1], "--gpu") == 0)
	{
		deviceSelection = 1;
	}
	else if (strcmp(argv[1], "--cpu") == 0)
	{
		deviceSelection = 0;
	}

	int n, w, h;
	double a = 100;
	char path[256];

	double r0c, z0c, p0c, arc, azc, wrc, wzc, wpc, prc, pzc,
		   r0n, z0n, p0n, arn, azn, wrn, wzn, wpn, prn, pzn;

	double r1 = 2, r2 = 2, r3 = 2;
	vec3 o1 = { -2, -2, 0 };
	vec3 o2 = { -2,  2, 0 };
	vec3 o3 = {  2,  0, 0 };

	int c1x, c1y, c1z;
	uchar4 c1 = { 200, 0, 0 };
	int c2x, c2y, c2z;
	uchar4 c2 = { 0, 255, 0 };
	int c3x, c3y, c3z;
	uchar4 c3 = { 0, 0, 255 };

	vec3 fv[4];
	fv[0] = { -5, -5, -3 };
	fv[1] = { -5,  5, -3 };
	fv[2] = {  5, -5, -3 };
	fv[3] = {  5,  5, -3 };

	int fcx, fcy, fcz;
	uchar4 fc = { 102, 62, 0 };

	scanf("%d\n", &n);
	scanf("%s\n", path);
	scanf("%d %d %lf\n", &w, &h, &a);

	scanf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", 
		&r0c, &z0c, &p0c, &arc, &azc, &wrc, &wzc, &wpc, &prc, &pzc);
	scanf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", 
		&r0n, &z0n, &p0n, &arn, &azn, &wrn, &wzn, &wpn, &prn, &pzn);

	scanf("%lf %lf %lf %lf %d %d %d\n", 
		&o1.x, &o1.y, &o1.z, &r1, &c1x, &c1y, &c1z);
	scanf("%lf %lf %lf %lf %d %d %d\n", 
		&o2.x, &o2.y, &o2.z, &r2, &c2x, &c2y, &c2z);
	scanf("%lf %lf %lf %lf %d %d %d\n", 
		&o3.x, &o3.y, &o3.z, &r3, &c3x, &c3y, &c3z);
	scanf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %d %d",    
		&fv[0].x, &fv[0].y, &fv[0].z, &fv[1].x, &fv[1].y, &fv[1].z,
		&fv[2].x, &fv[2].y, &fv[2].z, &fv[3].x, &fv[3].y, &fv[3].z,
		&fcx, &fcy, &fcz);

	c1.x = c1x;
	c1.y = c1y;
	c1.z = c1z;

	c2.x = c2x;
	c2.y = c2y;
	c2.z = c2z;

	c3.x = c3x;
	c3.y = c3y;
	c3.z = c3z;

	fc.x = fcx;
	fc.y = fcy;
	fc.z = fcz;

	char buff[256];
	uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	
	vec3 pc, pv;

	Triangle triangles[42];
	
	BuildStage(triangles, r1, o1, c1, r2, o2, c2, r3, o3, c3, fv, fc);

	double pi = acos(-1);

	double dt = 2 * pi / (double)n;

	vec3 lightPosition = { -2, 0, 4 };
	uchar4 lightColor = { 255, 255, 255 };

	float sharedTime = 0;

	if (deviceSelection == 0)
	{
		double cpuTime;

		for (int k = 0; k < n; ++k)
		{
			pc = CoordCameraFromTime(r0c, z0c, p0c, arc, azc, wrc, wzc, wpc, prc, pzc, k * dt);
			pv = CoordViewPointFromTime(r0n, z0n, p0n, arn, azn, wrn, wzn, wpn, prn, pzn, k * dt);
			
			auto start = steady_clock::now();

			Render(triangles, pc, pv, w, h, a, data, lightPosition, lightColor);
			
			auto end = steady_clock::now();

			cpuTime = ((double)duration_cast<microseconds>(end - start).count()) / 1000.0;
			
			sprintf(buff, path, k);
			printf("%s %e ms\n", buff, cpuTime);
			sharedTime += cpuTime;
			FILE* out = fopen(buff, "wb");
			
			fwrite(&w, sizeof(int), 1, out);
			fwrite(&h, sizeof(int), 1, out);
			fwrite(data, sizeof(uchar4), w * h, out);
			fclose(out);
		}
		printf("All time: %e ms\n", sharedTime);

	}
	else
	{
		float deviceTime = 0;
		cudaEvent_t start, stop;
		CSC(cudaEventCreate(&start));
		CSC(cudaEventCreate(&stop));

		Triangle* deviceTriangles;

		CSC(cudaMalloc((void**)(&deviceTriangles), 42 * sizeof(Triangle)));
		CSC(cudaMemcpy(deviceTriangles, triangles, 42 * sizeof(Triangle), cudaMemcpyHostToDevice));

		uchar4* deviceData;
		CSC(cudaMalloc((void**)(&deviceData), w * h * sizeof(uchar4)));

		for (int k = 0; k < n; ++k)
		{
			pc = CoordCameraFromTime(r0c, z0c, p0c, arc, azc, wrc, wzc, wpc, prc, pzc, k * dt);
			pv = CoordViewPointFromTime(r0n, z0n, p0n, arn, azn, wrn, wzn, wpn, prn, pzn, k * dt);
			
			CSC(cudaEventRecord(start));
			
			DeviceRender<<<128, 128>>>(deviceTriangles, pc, pv, w, h, a, deviceData, lightPosition, lightColor);
			CSC(cudaGetLastError());
			
			CSC(cudaEventRecord(stop));
			CSC(cudaEventSynchronize(stop));
			CSC(cudaEventElapsedTime(&deviceTime, start, stop));
			
			CSC(cudaMemcpy(data, deviceData, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost));

			sprintf(buff, path, k);
			printf("%s %e ms\n", buff, deviceTime);

			sharedTime += deviceTime;

			FILE* out = fopen(buff, "wb");

			fwrite(&w, sizeof(int), 1, out);
			fwrite(&h, sizeof(int), 1, out);
			fwrite(data, sizeof(uchar4), w * h, out);
			fclose(out);
		}
		printf("All time: %e ms\n", sharedTime);

		CSC(cudaEventDestroy(start));
		CSC(cudaEventDestroy(stop));
		CSC(cudaFree(deviceTriangles));
		CSC(cudaFree(deviceData));
	}
	free(data);
	return 0;
}