//////////////////////////////////////////////////////////////////////////////////
//                                                                           	//
//	cleap                                                                   //
//	A library for handling / processing / rendering 3D meshes.	        //
//                                                                           	//
//////////////////////////////////////////////////////////////////////////////////
//										//
//	Copyright © 2011 Cristobal A. Navarro.					//
//										//
//	This file is part of cleap.						//
//	cleap is free software: you can redistribute it and/or modify		//
//	it under the terms of the GNU General Public License as published by	//
//	the Free Software Foundation, either version 3 of the License, or	//
//	(at your option) any later version.					//
//										//
//	cleap is distributed in the hope that it will be useful,		//
//	but WITHOUT ANY WARRANTY; without even the implied warranty of		//
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the	    	//
//	GNU General Public License for more details.				//
//										//
//	You should have received a copy of the GNU General Public License	//
//	along with cleap.  If not, see <http://www.gnu.org/licenses/>. 		//
//										//
//////////////////////////////////////////////////////////////////////////////////

#ifndef _CLEAP_KERNEL_MOVE_VERTICES_H
#define _CLEAP_KERNEL_MOVE_VERTICES_H

#include "cleap_kernel_utils.cu"

__global__ void cleap_kernel_move_mesh(int vertex_count, float4* mesh_data, float2* vel, float2 min_p, float2 max_p){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<vertex_count && i>4){
        if(mesh_data[i].x>max_p.x || mesh_data[i].x<min_p.x)vel[i].x*=-1;
        if(mesh_data[i].y>max_p.y || mesh_data[i].y<min_p.y)vel[i].y*=-1;
        mesh_data[i].x+=vel[i].x;
        mesh_data[i].y+=vel[i].y;
    }
}

__global__ void cleap_kernel_correct_overlaps(unsigned int edge_count, GLuint* triangles, float4* mesh_data, int2* edge_idx, int* listo, float radius, float2* vel){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for(; i < edge_count; i += stride)
	{
		int2 edge = edge_idx[i];
		float4 bi = mesh_data[triangles[edge_idx[i].x]];
		float4 bj = mesh_data[triangles[edge_idx[i].y]];

		float2 r = distVec(bi,bj);
		double dist_sqrt = dot(r,r);

		if (dist_sqrt < 4*radius*radius)
		{
			// we have a collision
			listo[0] = 0;

			float2 p0 = make_float2(bi.x,bi.y);
			float2 v0 = vel[triangles[edge.x]];
			float2 p1 = make_float2(bj.x,bj.y);
			float2 v1 = vel[triangles[edge.y]];

			float2 contactNormal = normalize(p0-p1);
			float2 contactNormalNN = contactNormal*(2*radius-length(p0-p1));
			float separationVelocity = dot(v0-v1,contactNormal);
			float delta_velocity = -0.5*separationVelocity;
			float impulse = delta_velocity*2; // asumming mass = 1 for every particle
			float2 imp_per_mass = contactNormal*impulse;

			atomicAdd(&(vel[triangles[edge.x]].x),imp_per_mass.x);
			atomicAdd(&(vel[triangles[edge.x]].y),imp_per_mass.y);

			atomicAdd(&(vel[triangles[edge.y]].x),-imp_per_mass.x);
			atomicAdd(&(vel[triangles[edge.y]].y),-imp_per_mass.y);

			atomicAdd(&(mesh_data[triangles[edge.x]].x),contactNormalNN.x);
			atomicAdd(&(mesh_data[triangles[edge.x]].y),contactNormalNN.y);

			atomicAdd(&(mesh_data[triangles[edge.y]].x),-contactNormalNN.x);
			atomicAdd(&(mesh_data[triangles[edge.y]].y),-contactNormalNN.y);
		}
	}
}

#endif _CLEAP_KERNEL_MOVE_VERTICES_H
