var physicsShaderCode = `

struct Uniforms {
	time: f32,
	prevDeltaTime: f32,
	deltaTime: f32,
	gravity: f32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

//Can't use arrays of vec3f due to 16 byte padding
@group(0) @binding(1) var<storage, read> vertsPrev: array<f32>;
@group(0) @binding(2) var<storage, read_write> vertsOut: array<f32>;
@group(0) @binding(3) var<storage, read_write> normals: array<f32>;
@group(0) @binding(4) var<storage, read_write> accBuffer: array<atomic<i32>>;

const GRID_SIZE: u32 = 64;

fn readInputVtx(idx: u32) -> vec3f {
	return vec3f(
		vertsPrev[idx * 3    ],
		vertsPrev[idx * 3 + 1],
		vertsPrev[idx * 3 + 2]
	);
}

fn readOutputVtx(idx: u32) -> vec3f {
	return vec3f(
		vertsOut[idx * 3    ],
		vertsOut[idx * 3 + 1],
		vertsOut[idx * 3 + 2]
	);
}

const FP_PRECISION: f32 = 1 << 24; //no atomic floats in webgpu

fn readAccumCorrection(idx: u32) -> vec3f {
	return vec3f(vec3i(
		atomicLoad(&accBuffer[idx * 3    ]),
		atomicLoad(&accBuffer[idx * 3 + 1]),
		atomicLoad(&accBuffer[idx * 3 + 2])
	)) / FP_PRECISION;
}

fn isPointPinned(vtx: vec2u) -> bool {
	return 
		all(vtx == vec2u(0,             0            )) ||
		all(vtx == vec2u(GRID_SIZE - 1, 0            )) ||
		all(vtx == vec2u(GRID_SIZE - 1, GRID_SIZE - 1)) ||
		all(vtx == vec2u(0,             GRID_SIZE - 1)) ||
		
		all(vtx == vec2u(GRID_SIZE / 2, GRID_SIZE / 2))
	;
}

@compute @workgroup_size(8, 8)
fn addForces(
	@builtin(global_invocation_id) vtxPos: vec3u
) {
	//Update vertices positions with forces acceleration and other
	var vtxIdx: u32 = vtxPos.y * GRID_SIZE + vtxPos.x;
	
	//Set pinned points positions
	if(isPointPinned(vtxPos.xy)) {
		var tmpVec: vec3f = vec3f(vtxPos.xzy) / f32(GRID_SIZE - 1);
		
		//Move center point by sine
		if(all(vtxPos.xy == vec2u(GRID_SIZE / 2, GRID_SIZE / 2))) {
			tmpVec.y = sin(uniforms.time) * 0.5;
		}
		
		vertsOut[vtxIdx * 3    ] = tmpVec.x;
		vertsOut[vtxIdx * 3 + 1] = tmpVec.y;
		vertsOut[vtxIdx * 3 + 2] = tmpVec.z;
		
		return;
	}
	
	//Delta-time corrected verlet integration
	var vtx: vec3f = readInputVtx(vtxIdx); //old vtx value
	var vtxPrev: vec3f = readOutputVtx(vtxIdx); //pre old vtx
			
	var velocity: vec3f = vtx - vtxPrev;
	var damping: f32 = pow(0.99, uniforms.deltaTime * 60 * 20);
	
	vtx += velocity * damping * uniforms.deltaTime / uniforms.prevDeltaTime;
	
	//Apply acceleration (gravity)
	vtx.y -= uniforms.gravity * uniforms.deltaTime * (uniforms.deltaTime + uniforms.prevDeltaTime) / 2;
	
	//Sphere collision
	/*var spherePos: vec3f = vec3f(0.5 + sin(uniforms.time) * 2, -0.5, 0.5);
	var sphereDir: vec3f = vtx - spherePos;
	var dist: f32 = length(sphereDir);
	
	if(dist < 0.5f) {
		vtx = spherePos + sphereDir / dist * 0.5;
	}*/
	
	vertsOut[vtxIdx * 3    ] = vtx.x;
	vertsOut[vtxIdx * 3 + 1] = vtx.y;
	vertsOut[vtxIdx * 3 + 2] = vtx.z;
}


@compute @workgroup_size(8, 8)
fn resolveContraints(
	@builtin(global_invocation_id) vtxPos: vec3u
) {
	var vtxIdx: u32 = vtxPos.y * GRID_SIZE + vtxPos.x;
	var vtxPosi: vec2i = vec2i(vtxPos.xy);
	
	//Iterate over all connected UNPROCESSED springs
	// and accumulate corrections
	var vtx: vec3f = readOutputVtx(vtxIdx);
	
	const springLen: f32 = 1 / f32(GRID_SIZE - 1);
	
	//Neighbours are processed in such order
	// that only unprocessed springs are processed
	//
	// nnOyy
	//  yyy
	//   y
	for(var dz: i32 = 0; dz <= 2; dz++) {
		for(var dx: i32 = -2; dx <= 2; dx++) {
			//Skip already processed springs and current vert
			if(dx <= 0 && dz == 0) {continue;}
			
			//Skip out of bounds verts
			if(vtxPosi.x + dx < 0) {continue;}
			if(vtxPosi.y + dz < 0) {continue;}
			if(vtxPosi.x + dx >= i32(GRID_SIZE)) {continue;}
			if(vtxPosi.y + dz >= i32(GRID_SIZE)) {continue;}
			
			//Process only structure, shear and bend springs
			if(abs(dx) == 2 && abs(dz) == 2) {continue;}
			if(abs(dx) == 2 && abs(dz) == 1) {continue;}
			if(abs(dx) == 1 && abs(dz) == 2) {continue;}
			
			//Scale factor for non structure springs
			var scale: f32 = length(vec2f(f32(dx), f32(dz)));
			
			//Get connected neighbour position
			var neighIdx: u32 = u32(vtxPosi.x + dx + (vtxPosi.y + dz) * i32(GRID_SIZE));
			var neighVtx: vec3f = readOutputVtx(neighIdx);
			
			//Use updated positions
			var vtxTmp: vec3f = vtx + readAccumCorrection(vtxIdx);
			neighVtx += readAccumCorrection(neighIdx);
			
			//Calculate correction force from spring
			var dir: vec3f = neighVtx - vtxTmp;
			
			var targetLen: f32 = springLen * scale;
			var currentLen: f32 = length(dir);
			if(currentLen <= 0.0) {continue;}
			
			var compliance: f32 = 0.000005 * scale;
			var stiffness: f32 = 1 / (1 + compliance / uniforms.deltaTime / uniforms.deltaTime);
			
			var correctionForce: f32 = (currentLen - targetLen) * stiffness * 0.5f / currentLen;
			//float constraint = currentLen - targetLen;
			//float forceLen = -constraint / (1.0f + alpha); 
			
			var corrForce: vec3i = vec3i(dir * correctionForce * FP_PRECISION);
			var corrForceInv: vec3i = -corrForce;
			
			if(isPointPinned(vec2u(vtxPosi + vec2i(dx, dz)))) {
				corrForce *= 2;
				corrForceInv = vec3i(0);
			}
			
			if(isPointPinned(vtxPos.xy)) {
				corrForceInv *= 2;
				corrForce = vec3i(0);
			}
			
			//Add correction force to accumulation buffer
			atomicAdd(&accBuffer[vtxIdx * 3    ], corrForce.x);
			atomicAdd(&accBuffer[vtxIdx * 3 + 1], corrForce.y);
			atomicAdd(&accBuffer[vtxIdx * 3 + 2], corrForce.z);
			
			atomicAdd(&accBuffer[neighIdx * 3    ], corrForceInv.x);
			atomicAdd(&accBuffer[neighIdx * 3 + 1], corrForceInv.y);
			atomicAdd(&accBuffer[neighIdx * 3 + 2], corrForceInv.z);
		}
	}
}

@compute @workgroup_size(8, 8)
fn updateBuffers(
	@builtin(global_invocation_id) vtxPos: vec3u
) {
	//Add constraints corrections from accumulation buffer
	// and reset buffer
	var vtxIdx: u32 = vtxPos.y * GRID_SIZE + vtxPos.x;
	
	var force: vec3f = readAccumCorrection(vtxIdx);
	
	atomicStore(&accBuffer[vtxIdx * 3    ], 0);
	atomicStore(&accBuffer[vtxIdx * 3 + 1], 0);
	atomicStore(&accBuffer[vtxIdx * 3 + 2], 0);
	
	vertsOut[vtxIdx * 3    ] += force.x;
	vertsOut[vtxIdx * 3 + 1] += force.y;
	vertsOut[vtxIdx * 3 + 2] += force.z;
}

fn calcNormal(a: vec3f, b: vec3f, c: vec3f) -> vec3f {
    return normalize(cross(b - a, c - a));
}

@compute @workgroup_size(8, 8)
fn recalcNorms(
	@builtin(global_invocation_id) vtxPos: vec3u
) {
	//Calculate normals from 4 neighbour triangles to smooth normals a bit
	var vtxIdx: u32 = vtxPos.y * GRID_SIZE + vtxPos.x;
	var norm: vec3f = vec3f(0);
	
	if(vtxPos.x > 0 && vtxPos.y > 0) {
		var a: vec3f = readOutputVtx(vtxPos.x + vtxPos.y * GRID_SIZE);
		var b: vec3f = readOutputVtx(vtxPos.x + (vtxPos.y - 1) * GRID_SIZE);
		var c: vec3f = readOutputVtx(vtxPos.x - 1 + vtxPos.y * GRID_SIZE);

		norm += calcNormal(a, b, c);
	}

	if(vtxPos.x > 0 && vtxPos.y < GRID_SIZE - 1) {
		var a: vec3f = readOutputVtx(vtxPos.x + vtxPos.y * GRID_SIZE);
		var b: vec3f = readOutputVtx(vtxPos.x - 1 + vtxPos.y * GRID_SIZE);
		var c: vec3f = readOutputVtx(vtxPos.x + (vtxPos.y + 1) * GRID_SIZE);

		norm += calcNormal(a, b, c);
	}

	if(vtxPos.x < GRID_SIZE - 1 && vtxPos.y < GRID_SIZE - 1) {
		var a: vec3f = readOutputVtx(vtxPos.x + vtxPos.y * GRID_SIZE);
		var b: vec3f = readOutputVtx(vtxPos.x + (vtxPos.y + 1) * GRID_SIZE);
		var c: vec3f = readOutputVtx(vtxPos.x + 1 + vtxPos.y * GRID_SIZE);

		norm += calcNormal(a, b, c);
	}

	if(vtxPos.x < GRID_SIZE - 1 && vtxPos.y > 0) {
		var a: vec3f = readOutputVtx(vtxPos.x + vtxPos.y * GRID_SIZE);
		var b: vec3f = readOutputVtx(vtxPos.x + 1 + vtxPos.y * GRID_SIZE);
		var c: vec3f = readOutputVtx(vtxPos.x + (vtxPos.y - 1) * GRID_SIZE);

		norm += calcNormal(a, b, c);
	}
	
	norm = normalize(norm);

	normals[vtxIdx * 3    ] = norm.x;
	normals[vtxIdx * 3 + 1] = norm.y;
	normals[vtxIdx * 3 + 2] = norm.z;
}
`