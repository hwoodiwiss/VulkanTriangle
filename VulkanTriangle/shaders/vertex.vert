#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) in vec3 vertPos;
layout(location = 1) in vec4 vertColour;

layout(location = 0) out vec4 fragColor;


void main() {
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(vertPos, 1.0);
    fragColor = vertColour;
}