shader_type canvas_item;

uniform float time;
uniform vec4 color1 : source_color = vec4(0.1, 0.15, 0.25, 1.0);
uniform vec4 color2 : source_color = vec4(0.2, 0.25, 0.4, 1.0);
uniform vec4 color3 : source_color = vec4(0.15, 0.2, 0.35, 1.0);
uniform float wave_speed : hint_range(0.1, 2.0) = 0.5;
uniform float wave_frequency : hint_range(1.0, 10.0) = 3.0;

void fragment() {
	vec2 uv = UV;
	
	// Create flowing gradient effect
	float wave1 = sin(uv.x * wave_frequency + time * wave_speed) * 0.5 + 0.5;
	float wave2 = cos(uv.y * wave_frequency - time * wave_speed * 0.7) * 0.5 + 0.5;
	float wave3 = sin((uv.x + uv.y) * wave_frequency * 0.5 + time * wave_speed * 0.5) * 0.5 + 0.5;
	
	// Mix colors based on waves
	vec4 mixed_color = mix(color1, color2, wave1);
	mixed_color = mix(mixed_color, color3, wave2 * wave3);
	
	// Add subtle noise
	float noise = fract(sin(dot(uv + time * 0.1, vec2(12.9898, 78.233))) * 43758.5453);
	mixed_color.rgb += noise * 0.02;
	
	COLOR = mixed_color;
}
