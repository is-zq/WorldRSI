shader_type spatial;
render_mode unshaded, fog_disabled;
uniform float instance_code;
uniform bool normals = false;
varying vec3 world_normal;
void vertex() {
    world_normal = MODEL_NORMAL_MATRIX * NORMAL;
}
void fragment() {
    vec3 digits = vec3(floor(instance_code / 16384.0), mod(floor(instance_code / 128.0), 128.0), mod(instance_code, 128.0));
    if (normals) {
        digits = round((normalize(world_normal) * 0.5 + 0.5) * 127.0);
    }
    ALBEDO = (digits + 64.0) / 255.0;
}
