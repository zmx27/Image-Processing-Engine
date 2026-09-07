#include "imgjit/core/op_chain.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace imgjit {
namespace {

std::string_view trim(std::string_view text) {
  const auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
  while (!text.empty() && is_space(text.front())) {
    text.remove_prefix(1);
  }
  while (!text.empty() && is_space(text.back())) {
    text.remove_suffix(1);
  }
  return text;
}

const OpSpec* find_spec(std::string_view name) {
  for (const OpSpec& spec : kOpSpecs) {
    if (spec.name == name) {
      return &spec;
    }
  }
  return nullptr;
}

bool fail(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
  return false;
}

// strtof rather than from_chars: libc++ shipped floating-point from_chars late, and
// this has to build on whatever AppleClang the Mac has. endptr gives us the "whole
// token consumed" check that makes `1.4x` a parse error rather than 1.4.
bool parse_float(std::string_view text, float* out) {
  const std::string token(text);
  if (token.empty()) {
    return false;
  }
  char* end = nullptr;
  const float value = std::strtof(token.c_str(), &end);
  if (end != token.c_str() + token.size() || !std::isfinite(value)) {
    return false;
  }
  *out = value;
  return true;
}

// Two decimals, trailing zeros stripped, but never a bare integer: 1.00 -> "1.0",
// 1.40 -> "1.4", 1.45 -> "1.45". Printed from the quantized value, so the text and the
// hashed number can never disagree.
std::string format_param(float value) {
  const long scaled = std::lround(static_cast<double>(value) * 100.0);
  const long magnitude = std::labs(scaled);
  std::string text = (scaled < 0 ? std::string("-") : std::string("")) +
                     std::to_string(magnitude / 100) + "." +
                     std::to_string((magnitude / 10) % 10);
  const long hundredths = magnitude % 10;
  if (hundredths != 0) {
    text += std::to_string(hundredths);
  }
  return text;
}

}  // namespace

const OpSpec& op_spec(OpKind kind) {
  const auto index = static_cast<std::size_t>(kind);
  if (index >= kOpSpecs.size() || kOpSpecs[index].kind != kind) {
    throw std::invalid_argument("op_spec: unknown OpKind");
  }
  return kOpSpecs[index];
}

float quantize_param(float value) {
  static_assert(kParamDecimals == 2, "format_param and the KernelKey encoding assume 2 decimals");
  return static_cast<float>(std::lround(static_cast<double>(value) * 100.0)) / 100.0F;
}

std::optional<OpChain> parse_op_chain(std::string_view text, std::string* error) {
  OpChain chain;
  const std::string_view body = trim(text);
  if (body.empty()) {
    return chain;
  }

  std::size_t position = 0;
  while (position <= body.size()) {
    const std::size_t comma = body.find(',', position);
    const std::string_view token =
        trim(body.substr(position, comma == std::string_view::npos ? comma : comma - position));

    if (token.empty()) {
      fail(error, "empty op in chain");
      return std::nullopt;
    }

    const std::size_t colon = token.find(':');
    const std::string_view name = trim(token.substr(0, colon));
    const OpSpec* spec = find_spec(name);
    if (spec == nullptr) {
      fail(error, "unknown op '" + std::string(name) + "'");
      return std::nullopt;
    }

    Op op{spec->kind, spec->default_param};
    if (colon != std::string_view::npos) {
      if (!spec->takes_param) {
        fail(error, "op '" + std::string(name) + "' takes no parameter");
        return std::nullopt;
      }
      float value = 0.0F;
      const std::string_view parameter = trim(token.substr(colon + 1));
      if (!parse_float(parameter, &value)) {
        fail(error, "op '" + std::string(name) + "': '" + std::string(parameter) +
                        "' is not a number");
        return std::nullopt;
      }
      value = quantize_param(value);
      if (value < spec->min_param || value > spec->max_param) {
        fail(error, "op '" + std::string(name) + "': parameter " + format_param(value) +
                        " is outside [" + format_param(spec->min_param) + ", " +
                        format_param(spec->max_param) + "]");
        return std::nullopt;
      }
      op.param = value;
    }
    chain.ops.push_back(op);

    if (comma == std::string_view::npos) {
      break;
    }
    position = comma + 1;
  }

  return chain;
}

std::string canonical_string(const OpChain& chain) {
  std::string text;
  for (const Op& op : chain.ops) {
    if (!text.empty()) {
      text += ',';
    }
    const OpSpec& spec = op_spec(op.kind);
    text += spec.name;
    if (spec.takes_param) {
      text += ':';
      text += format_param(op.param);
    }
  }
  return text;
}

}  // namespace imgjit
