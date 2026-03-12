#pragma once
#include <ATen/ATen.h>
#include <vector>

std::vector<at::Tensor> nms_forward(
    at::Tensor boxes,
    at::Tensor scores,
    float thresh,
    unsigned long top_k);