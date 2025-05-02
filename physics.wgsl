var physicsShaderCode = `

struct Uniforms {
	time: f32,
	prevDeltaTime: f32,
	deltaTime: f32,
	gravity: f32,
	stretchConfliance: f32,
	bendConfliance: f32,
	pinPointsMask: i32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

//Can't use arrays of vec3f due to 16 byte padding
@group(0) @binding(1) var<storage, read> vertsPrev: array<f32>;
@group(0) @binding(2) var<storage, read_write> vertsOut: array<f32>;
@group(0) @binding(3) var<storage, read_write> normals: array<f32>;
@group(0) @binding(4) var<storage, read_write> accBuffer: array<atomic<i32>>;

const GRID_SIZE: i32 = 64;

fn readInputVtx(idx: i32) -> vec3f {
	return vec3f(
		vertsPrev[idx * 3    ],
		vertsPrev[idx * 3 + 1],
		vertsPrev[idx * 3 + 2]
	);
}

fn readOutputVtx(idx: i32) -> vec3f {
	return vec3f(
		vertsOut[idx * 3    ],
		vertsOut[idx * 3 + 1],
		vertsOut[idx * 3 + 2]
	);
}

const FP_PRECISION: f32 = 1 << 24; //no atomic floats in webgpu

fn readAccumCorrection(idx: i32) -> vec3f {
	return vec3f(vec3i(
		atomicLoad(&accBuffer[idx * 3    ]),
		atomicLoad(&accBuffer[idx * 3 + 1]),
		atomicLoad(&accBuffer[idx * 3 + 2])
	)) / FP_PRECISION;
}

fn isPointPinned(vtx: vec2i) -> bool {
	return 
		(all(vtx == vec2i(0,             0            )) && (uniforms.pinPointsMask & 1) != 0) ||
		(all(vtx == vec2i(GRID_SIZE - 1, 0            )) && (uniforms.pinPointsMask & 2) != 0) ||
		(all(vtx == vec2i(GRID_SIZE - 1, GRID_SIZE - 1)) && (uniforms.pinPointsMask & 4) != 0) ||
		(all(vtx == vec2i(0,             GRID_SIZE - 1)) && (uniforms.pinPointsMask & 8) != 0) ||
		
		(all(vtx == vec2i(GRID_SIZE / 2, GRID_SIZE / 2)) && (uniforms.pinPointsMask & 16) != 0)
	;
}

@compute @workgroup_size(8, 8)
fn addForces(
	@builtin(global_invocation_id) vtxPos: vec3u
) {
	//Update vertices positions with forces acceleration and other
	var vtxPosi: vec2i = vec2i(vtxPos.xy);
	var vtxIdx: i32 = vtxPosi.y * GRID_SIZE + vtxPosi.x;
	
	//Set pinned points positions
	if(isPointPinned(vtxPosi)) {
		var tmpVec: vec3f = vec3f(f32(vtxPosi.x), 0, f32(vtxPosi.y)) / f32(GRID_SIZE - 1);
		
		//Move center point by sine
		if(all(vtxPosi == vec2i(GRID_SIZE / 2, GRID_SIZE / 2))) {
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

fn processBendConstraint(
	v1: vec2i, v2: vec2i, v3: vec2i, v4: vec2i
) {
	//Read actual vertices data
	var v1Idx: i32 = v1.y * GRID_SIZE + v1.x;
	var v2Idx: i32 = v2.y * GRID_SIZE + v2.x;
	var v3Idx: i32 = v3.y * GRID_SIZE + v3.x;
	var v4Idx: i32 = v4.y * GRID_SIZE + v4.x;
	
	var p1: vec3f = readOutputVtx(v1Idx) + readAccumCorrection(v1Idx);
	var p2: vec3f = readOutputVtx(v2Idx) + readAccumCorrection(v2Idx);
	var p3: vec3f = readOutputVtx(v3Idx) + readAccumCorrection(v3Idx);
	var p4: vec3f = readOutputVtx(v4Idx) + readAccumCorrection(v4Idx);
	
	//Get inv masses
	var w1: f32 = f32(!isPointPinned(v1));
	var w2: f32 = f32(!isPointPinned(v2));
	var w3: f32 = f32(!isPointPinned(v3));
	var w4: f32 = f32(!isPointPinned(v4));
	
	//Set p1 to 0, 0
	p2 -= p1;
	p3 -= p1;
	p4 -= p1;
	
	//Precalculate cross products and normals
	var p2p3c: vec3f = cross(p2, p3);
	var p2p4c: vec3f = cross(p2, p4);
	
	var p2p3cLen: f32 = length(p2p3c);
	var p2p4cLen: f32 = length(p2p4c);
	
	const eps: f32 = 0.00000001;
	if(p2p3cLen < eps || p2p4cLen < eps) {return;}
	
	var n1: vec3f = p2p3c / p2p3cLen;
	var n2: vec3f = p2p4c / p2p4cLen;
	
	var d: f32 = clamp(dot(n1, n2), -1, 1);
	var currentAngle: f32 = acos(d);
	
	//Compute q(s)
	var q3: vec3f = (cross(p2, n2) + cross(n1, p2) * d) / p2p3cLen;
	var q4: vec3f = (cross(p2, n1) + cross(n2, p2) * d) / p2p4cLen;
	
	var q2: vec3f =
		-(cross(p3, n2) + cross(n1, p3) * d) / p2p3cLen
		-(cross(p4, n1) + cross(n2, p4) * d) / p2p4cLen;
		
	var q1: vec3f = -q2 - q3 - q4;
	
	//Compute final corrections
	const phi: f32 = radians(180);
	
	var scale: f32 = -sqrt(1 - d*d) * (currentAngle - phi);
	var denom: f32 = w1 * dot(q1, q1) + w2 * dot(q2, q2) + w3 * dot(q3, q3) + w4 * dot(q4, q4);
	
	//Cloth explodes if stiffness is too high
	var compliance: f32 = uniforms.bendConfliance;
	var invStiff = max(2.5, 1 + compliance / uniforms.deltaTime / uniforms.deltaTime);
	
	denom *= invStiff;
	
	if(denom < eps) {return;}
	scale /= denom;
	
	var dp1: vec3i = vec3i(q1 * w1 * scale * FP_PRECISION);
	var dp2: vec3i = vec3i(q2 * w2 * scale * FP_PRECISION);
	var dp3: vec3i = vec3i(q3 * w3 * scale * FP_PRECISION);
	var dp4: vec3i = vec3i(q4 * w4 * scale * FP_PRECISION);
	
	//Store deltas
	atomicAdd(&accBuffer[v1Idx * 3    ], dp1.x);
	atomicAdd(&accBuffer[v1Idx * 3 + 1], dp1.y);
	atomicAdd(&accBuffer[v1Idx * 3 + 2], dp1.z);
	
	atomicAdd(&accBuffer[v2Idx * 3    ], dp2.x);
	atomicAdd(&accBuffer[v2Idx * 3 + 1], dp2.y);
	atomicAdd(&accBuffer[v2Idx * 3 + 2], dp2.z);
	
	atomicAdd(&accBuffer[v3Idx * 3    ], dp3.x);
	atomicAdd(&accBuffer[v3Idx * 3 + 1], dp3.y);
	atomicAdd(&accBuffer[v3Idx * 3 + 2], dp3.z);
	
	atomicAdd(&accBuffer[v4Idx * 3    ], dp4.x);
	atomicAdd(&accBuffer[v4Idx * 3 + 1], dp4.y);
	atomicAdd(&accBuffer[v4Idx * 3 + 2], dp4.z);
}

@compute @workgroup_size(8, 8)
fn resolveContraints(
	@builtin(global_invocation_id) vtxPos: vec3u
) {
	var vtxPosi: vec2i = vec2i(vtxPos.xy);
	var vtxIdx: i32 = vtxPosi.y * GRID_SIZE + vtxPosi.x;
	
	//Iterate over all connected UNPROCESSED springs
	// and accumulate corrections
	var vtx: vec3f = readOutputVtx(vtxIdx);
	
	const springLen: f32 = 1 / f32(GRID_SIZE - 1);
	
	//Distance constraints (springs)
	// Neighbours are processed in such order
	// that only unprocessed springs are processed
	//
	//  Oy
	// yyy
	for(var dz: i32 = 0; dz <= 1; dz++) {
		for(var dx: i32 = -1; dx <= 1; dx++) {
			//Skip processed springs and current vert
			if(dx <= 0 && dz == 0) {continue;}
			
			//Skip out of bounds verts
			if(vtxPosi.x + dx < 0) {continue;}
			if(vtxPosi.x + dx >= GRID_SIZE) {continue;}
			if(vtxPosi.y + dz >= GRID_SIZE) {continue;}
			
			//Scale factor for non structure springs
			var diagScale: f32 = length(vec2f(f32(dx), f32(dz)));
			
			//Get connected neighbour position
			var neighIdx: i32 = vtxPosi.x + dx + (vtxPosi.y + dz) * GRID_SIZE;
			var p2: vec3f = readOutputVtx(neighIdx);
			
			//Use updated positions
			var p1: vec3f = vtx + readAccumCorrection(vtxIdx);
			p2 += readAccumCorrection(neighIdx);
			
			//Calculate correction force from spring
			var dir: vec3f = p1 - p2;
			
			var targetLen: f32 = springLen * diagScale;
			var currentLen: f32 = length(dir);
			if(currentLen <= 0.0) {continue;}
			
			var compliance: f32 = uniforms.stretchConfliance * diagScale;
			var stiffness: f32 = 1 / (1 + compliance / uniforms.deltaTime / uniforms.deltaTime);
			
			var scale: f32 = (currentLen - targetLen) * stiffness * 0.5f / currentLen;
			
			var corrForce: vec3i = vec3i(dir * -scale * FP_PRECISION);
			var corrForceInv: vec3i = -corrForce;
			
			if(isPointPinned(vtxPosi + vec2i(dx, dz))) {
				corrForce *= 2;
				corrForceInv = vec3i(0);
			}
			
			if(isPointPinned(vtxPosi)) {
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
	
	//Bend constraints
	if(vtxPosi.x + 1 < GRID_SIZE && vtxPosi.y + 1 < GRID_SIZE) {
		processBendConstraint(
			vtxPosi, 
			vtxPosi + vec2i(1, 1),
			vtxPosi + vec2i(1, 0),
			vtxPosi + vec2i(0, 1),
		);
		
		//Actual mesh structure is different but using actual mesh structure
		// leads to asymmetry when constraints are applied
		if(vtxPosi.x + 2 < GRID_SIZE) {
			processBendConstraint(
				vtxPosi + vec2i(1, 1), 
				vtxPosi + vec2i(1, 0),
				vtxPosi,
				vtxPosi + vec2i(2, 0),
			);
		}

		if(vtxPosi.y + 2 < GRID_SIZE) {
			processBendConstraint(
				vtxPosi + vec2i(0, 1), 
				vtxPosi + vec2i(1, 1),
				vtxPosi,
				vtxPosi + vec2i(0, 2),
			);
		}
	}
}

@compute @workgroup_size(8, 8)
fn updateBuffers(
	@builtin(global_invocation_id) vtxPos: vec3u
) {
	//Add constraints corrections from accumulation buffer
	// and reset buffer
	var vtxIdx: i32 = i32(vtxPos.y) * GRID_SIZE + i32(vtxPos.x);
	
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
	var vtxPosi: vec2i = vec2i(vtxPos.xy);
	var vtxIdx: i32 = vtxPosi.y * GRID_SIZE + vtxPosi.x;
	
	var norm: vec3f = vec3f(0);
	
	if(vtxPosi.x > 0 && vtxPosi.y > 0) {
		var a: vec3f = readOutputVtx(vtxPosi.x + vtxPosi.y * GRID_SIZE);
		var b: vec3f = readOutputVtx(vtxPosi.x + (vtxPosi.y - 1) * GRID_SIZE);
		var c: vec3f = readOutputVtx(vtxPosi.x - 1 + vtxPosi.y * GRID_SIZE);

		norm += calcNormal(a, b, c);
	}

	if(vtxPosi.x > 0 && vtxPosi.y < GRID_SIZE - 1) {
		var a: vec3f = readOutputVtx(vtxPosi.x + vtxPosi.y * GRID_SIZE);
		var b: vec3f = readOutputVtx(vtxPosi.x - 1 + vtxPosi.y * GRID_SIZE);
		var c: vec3f = readOutputVtx(vtxPosi.x + (vtxPosi.y + 1) * GRID_SIZE);

		norm += calcNormal(a, b, c);
	}

	if(vtxPosi.x < GRID_SIZE - 1 && vtxPosi.y < GRID_SIZE - 1) {
		var a: vec3f = readOutputVtx(vtxPosi.x + vtxPosi.y * GRID_SIZE);
		var b: vec3f = readOutputVtx(vtxPosi.x + (vtxPosi.y + 1) * GRID_SIZE);
		var c: vec3f = readOutputVtx(vtxPosi.x + 1 + vtxPosi.y * GRID_SIZE);

		norm += calcNormal(a, b, c);
	}

	if(vtxPosi.x < GRID_SIZE - 1 && vtxPosi.y > 0) {
		var a: vec3f = readOutputVtx(vtxPosi.x + vtxPosi.y * GRID_SIZE);
		var b: vec3f = readOutputVtx(vtxPosi.x + 1 + vtxPosi.y * GRID_SIZE);
		var c: vec3f = readOutputVtx(vtxPosi.x + (vtxPosi.y - 1) * GRID_SIZE);

		norm += calcNormal(a, b, c);
	}
	
	norm = normalize(norm);

	normals[vtxIdx * 3    ] = norm.x;
	normals[vtxIdx * 3 + 1] = norm.y;
	normals[vtxIdx * 3 + 2] = norm.z;
}
`