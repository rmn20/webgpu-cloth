var renderShaderCode = `

struct Uniforms {
	viewMat: mat4x4f,
	projMat: mat4x4f,
	viewPos: vec4f
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
	@builtin(position) pos: vec4f,
	@location(0) fragPos: vec3f,
	@location(1) fragNorm: vec3f,
};

@vertex
fn vertexMain(
	@location(0) position: vec3f,
	@location(1) normal: vec3f,
) -> VertexOutput {
	var out: VertexOutput;
	
	out.pos = uniforms.projMat * (uniforms.viewMat * vec4f(position, 1));
	out.fragPos = position;
	out.fragNorm = normal;

	return out;
}


fn hash(p: vec2f) -> f32 { return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); }

@fragment
fn fragmentMain(
	@builtin(front_facing) frontFacing : bool,
	input: VertexOutput
) -> @location(0) vec4f {
    var texelColor: vec4f = vec4f(1);

	//Lighting
    var normal: vec3f = normalize(input.fragNorm);
	if(!frontFacing) {normal *= -1;}
	
	var lightDir: vec3f = normalize(vec3f(-1, -1.4, -1));
    var viewDir: vec3f = normalize(uniforms.viewPos.xyz - input.fragPos);
	var halfVec: vec3f = normalize(viewDir - lightDir);
	
	var light: vec3f = mix(vec3f(0.3), vec3f(1.0), max(0.0, -dot(normal, lightDir)));
	var specular: vec3f = vec3f(0.3) * pow(max(0.0, dot(halfVec, normal)), 24.0);
	var rim: vec3f = vec3f(0.3) * pow((1.0 - max(0.0, dot(viewDir, normal))), 2.4);
	
	//Apply lighting in approximate linear rgb color space
	var resultCol = texelColor.rgb;
	
	resultCol = pow(resultCol.rgb, vec3f(2.2));
    resultCol = (resultCol.rgb * light + specular + rim) * 0.22;
	resultCol = pow(resultCol.rgb, vec3f(1.0 / 2.2));
	
	//Dithering
	resultCol += vec3f(hash(input.pos.xy)) / 255.0;
	
	return vec4f(resultCol, texelColor.a);
}
`