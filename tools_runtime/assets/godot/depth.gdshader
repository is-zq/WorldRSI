shader_type spatial;
render_mode unshaded, fog_disabled, cull_disabled, depth_draw_never, depth_test_disabled;

uniform sampler2D depth_texture : hint_depth_texture, filter_nearest;
uniform float far_distance;

void vertex() {
    POSITION = vec4(VERTEX.xy, 1.0, 1.0);
}

void fragment() {
    float raw = textureLod(depth_texture, SCREEN_UV, 0.0).r;
    // Compatibility uses OpenGL NDC; Godot 4.3+ uses reversed Z.
    vec3 ndc = vec3(SCREEN_UV, raw) * 2.0 - 1.0;
    vec4 view = INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
    float z = -view.z / view.w;
    // Base-128 digits use bytes 64..191, where Compatibility's approximate sRGB
    // round trip preserves every byte. This transports 21 bits without dark-byte loss.
    // Zero denotes background; decoded depth is camera Z in scene units.
    float packed = raw > 0.0 ? clamp(round(z / far_distance * 2097151.0), 1.0, 2097151.0) : 0.0;
    ALBEDO = (vec3(floor(packed / 16384.0), mod(floor(packed / 128.0), 128.0), mod(packed, 128.0)) + 64.0) / 255.0;
    ALPHA = 1.0;
}
