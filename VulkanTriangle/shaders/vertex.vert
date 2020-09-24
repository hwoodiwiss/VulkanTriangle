#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vertPos;
layout(location = 1) in vec4 vertColour;

layout(location = 0) out vec4 fragColor;


void main() {
    gl_Position = vec4(vertPos, 1.0);
    fragColor = vertColour;
}