#include "backend/cuda/codegen.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/cuda/kernel_prelude.h"
#include "imgjit/backend/cpu/ops.h"
#include "imgjit/core/image.h"

namespace imgjit::cuda {
namespace {

constexpr const char* kStagePrefix = "imgjit_stage";

// Static __shared__ memory per block that every CUDA device since compute 2.0
// guarantees. The closed op set's worst case — tile 32 and gaussian:4 (radius 12), so a
// 56x56 apron, times three colour planes of floats (alpha is never staged) — is 37,632
// bytes, so no valid key reaches this. It is checked anyway because the failure it
// prevents is not local: an oversized tile compiles to PTX and only fails when the
// driver loads the module, which is a CudaError, which is a context recreation on every
// frame instead of an error naming the tile.
constexpr std::size_t kMaxStaticSharedBytes = 48 * 1024;

// %.9g is the shortest precision that round-trips every float exactly, so a literal
// printed here parses back to the bit pattern the oracle computed. That matters most
// for the gaussian weights: they are read straight out of the CPU implementation, and
// a lossy print would make the two convolve with different numbers.
std::string float_literal(float value) {
  std::array<char, 32> text{};
  std::snprintf(text.data(), text.size(), "%.9g", static_cast<double>(value));
  std::string literal(text.data());
  if (literal.find_first_of(".eE") == std::string::npos) {
    literal += ".0";
  }
  return literal + "f";
}

// An op together with its position in the chain. The position names a parameterized
// op's kernel argument (`p<index>`) and is what GeneratedStage::param_ops reports.
struct PlannedOp {
  Op op{};
  std::size_t index{0};
};

// One kernel's worth of the chain: the stencil it is built around, plus the pointwise
// runs folded in before and after it. A stage with no stencil is a pure pointwise
// kernel, and only a chain containing no stencil at all produces one — "gaussian,
// invert" folds the invert into the gaussian's epilogue rather than emitting a second
// kernel for it.
struct Stage {
  std::vector<PlannedOp> prologue;  // applied at every stencil tap, before the stencil
  bool has_stencil{false};
  PlannedOp stencil{};
  std::vector<PlannedOp> epilogue;  // applied once, to the stencil's result
};

std::vector<Stage> plan_stages(const OpChain& chain) {
  std::vector<Stage> stages;
  for (std::size_t index = 0; index < chain.ops.size(); ++index) {
    const PlannedOp planned{chain.ops[index], index};
    if (stages.empty()) {
      stages.emplace_back();
    }
    if (!op_spec(planned.op.kind).is_stencil) {
      Stage& current = stages.back();
      // Prefer attaching backwards: an epilogue runs once per output pixel, a
      // prologue runs once per tap.
      (current.has_stencil ? current.epilogue : current.prologue).push_back(planned);
      continue;
    }
    if (stages.back().has_stencil) {
      stages.emplace_back();
    }
    stages.back().has_stencil = true;
    stages.back().stencil = planned;
  }
  return stages;
}

// The chain indices of the pointwise ops in `ops` whose parameter is a kernel argument.
// None when baked, where every parameter is a literal instead.
std::vector<std::size_t> param_indices(const std::vector<PlannedOp>& ops, bool parameterized) {
  std::vector<std::size_t> indices;
  if (parameterized) {
    for (const PlannedOp& planned : ops) {
      if (op_spec(planned.op.kind).takes_param) {
        indices.push_back(planned.index);
      }
    }
  }
  return indices;
}

// `prefix` is ", float p" to declare the arguments, or ", p" to pass them on.
std::string param_list(const std::vector<std::size_t>& indices, const char* prefix) {
  std::string text;
  for (const std::size_t index : indices) {
    text += prefix + std::to_string(index);
  }
  return text;
}

// The chain as the header comment names it. A parameterized source must not depend on
// any parameter value — every value shares its cache entry, so a printed sigma would be
// whichever frame compiled it first — so the values are elided.
std::string chain_comment(const OpChain& chain, bool parameterized) {
  if (!parameterized) {
    return canonical_string(chain);
  }
  std::string text;
  for (const Op& op : chain.ops) {
    text += text.empty() ? "" : ",";
    text += op_spec(op.kind).name;
    text += op_spec(op.kind).takes_param ? ":<arg>" : "";
  }
  return text;
}

// Emits a pointwise run over the register array `var[0 .. colors)`.
//
// Every op ends in imgjit_quantize, without exception. See the comment on that helper
// in kernel_prelude.h: the oracle stores a uint8 image between ops, so a fused kernel
// that stayed in float would not be running the same filter. `invert` is written in
// float here rather than as integer 255-u and still matches the oracle exactly — the
// true result is an integer and the float error is ~1e-4, far inside the rounding.
void emit_pointwise(std::ostringstream& out, const std::vector<PlannedOp>& ops, int colors,
                    const char* var, const char* indent, bool parameterized) {
  for (const PlannedOp& planned : ops) {
    const Op& op = planned.op;
    // A literal when baked; the kernel argument carrying this op's value when not.
    const std::string param =
        parameterized ? "p" + std::to_string(planned.index) : float_literal(op.param);
    switch (op.kind) {
      case OpKind::kGrayscale:
        if (colors == 1) {
          out << indent << "// grayscale: a 1-channel sample is already luminance.\n";
          break;
        }
        out << indent << "{\n"
            << indent << "  const float g = imgjit_quantize(imgjit_luma(" << var << "[0], " << var
            << "[1], " << var << "[2]));\n";
        for (int c = 0; c < colors; ++c) {
          out << indent << "  " << var << "[" << c << "] = g;\n";
        }
        out << indent << "}\n";
        break;
      case OpKind::kInvert:
        for (int c = 0; c < colors; ++c) {
          out << indent << var << "[" << c << "] = imgjit_quantize(1.0f - " << var << "[" << c
              << "]);\n";
        }
        break;
      case OpKind::kBrightness:
        for (int c = 0; c < colors; ++c) {
          out << indent << var << "[" << c << "] = imgjit_quantize(" << var << "[" << c << "] + "
              << param << ");\n";
        }
        break;
      case OpKind::kThreshold:
        for (int c = 0; c < colors; ++c) {
          out << indent << var << "[" << c << "] = imgjit_quantize(" << var << "[" << c
              << "] >= " << param << " ? 1.0f : 0.0f);\n";
        }
        break;
      case OpKind::kGaussian:
      case OpKind::kSobel:
        throw std::logic_error("emit_pointwise: a stencil op reached a pointwise run");
    }
  }
}

std::string zero_initializer(int colors) {
  std::string text = "{";
  for (int c = 0; c < colors; ++c) {
    text += (c == 0 ? "0.0f" : ", 0.0f");
  }
  return text + "}";
}

// The stage's neighbour accessor: one clamped (replicate) load with the prologue
// folded in. Emitted only for stencil stages — it is the thing the stencil's tap loop
// calls, and folding the prologue in here is what "pointwise ops fuse at zero memory
// cost" means concretely.
// Parameterized, the prologue's values arrive as trailing `float p<index>` arguments.
void emit_tap_function(std::ostringstream& out, const std::string& name, const Stage& stage,
                       int channels, int colors, bool parameterized,
                       const std::string& prologue_params) {
  out << "\n__device__ __forceinline__ void " << name << "_tap(\n"
      << "    const unsigned char* __restrict__ src, int x, int y, int width, int height,\n"
      << "    float* v" << prologue_params << ") {\n"
      << "  const int sx = imgjit_clamp_coord(x, width);\n"
      << "  const int sy = imgjit_clamp_coord(y, height);\n"
      << "  const int i = (sy * width + sx) * " << channels << ";\n";
  for (int c = 0; c < colors; ++c) {
    out << "  v[" << c << "] = imgjit_load_sample(src[i + " << c << "]);\n";
  }
  emit_pointwise(out, stage.prologue, colors, "v", "  ", parameterized);
  out << "}\n";
}

// Baked stencil constants. Radius and weights are literals; width and height never
// are (invariant 4). Returns the radius, which the kernel body needs.
int emit_stencil_constants(std::ostringstream& out, const std::string& name, const Op& stencil) {
  if (stencil.kind == OpKind::kSobel) {
    out << "\n__device__ const float " << name
        << "_gx[9] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};\n"
        << "__device__ const float " << name
        << "_gy[9] = {-1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f};\n";
    return 1;
  }

  const int radius = cpu::gaussian_radius(stencil.param);
  const std::vector<float> weights = cpu::gaussian_weights_1d(stencil.param);
  // Read out of the CPU implementation rather than recomputed here, so the oracle and
  // the kernel convolve with bit-identical numbers instead of two expf results that
  // happen to agree.
  out << "\n// gaussian sigma=" << float_literal(stencil.param) << ", radius=" << radius
      << " — weights baked from imgjit::cpu::gaussian_weights_1d.\n"
      << "__device__ const float " << name << "_w[" << weights.size() << "] = {";
  for (std::size_t i = 0; i < weights.size(); ++i) {
    out << (i == 0 ? "" : ", ") << float_literal(weights[i]);
  }
  out << "};\n";
  return radius;
}

// The tiled variant's load phase (docs/PLAN.md Phase 7). The block stages the input for
// its tile_size x tile_size outputs plus a radius-wide apron on every side: that is
// (tile_size + 2*radius)^2 samples, more than the block has threads, so thread k loads
// cells k, k + tile_size^2, k + 2*tile_size^2, ... until the tile is full.
//
// The load goes through the stage's tap helper — the naive kernel's own clamped
// (replicate) load with the prologue folded in — so edge handling is one piece of code
// in both variants, and the prologue now runs once per staged sample instead of once
// per tap.
//
// The layout is one float plane per colour channel, so consecutive threads read
// consecutive words. Floats rather than bytes because what is staged is the value
// AFTER conversion and prologue; storing bytes would put a divide back into every tap.
void emit_tile_load(std::ostringstream& out, const std::string& name, int radius,
                    int tile_size, int colors) {
  const int edge = tile_size + 2 * radius;
  const std::size_t shared_bytes = static_cast<std::size_t>(colors * edge * edge) * sizeof(float);
  if (shared_bytes > kMaxStaticSharedBytes) {
    throw std::invalid_argument("emit_cuda_source: a " + std::to_string(tile_size) +
                                "-wide tile with a radius-" + std::to_string(radius) +
                                " apron needs " + std::to_string(shared_bytes) +
                                " bytes of __shared__ memory, over the 48 KiB static limit");
  }

  out << "  // " << tile_size << "x" << tile_size << " outputs plus a " << radius
      << "-pixel apron on every side, staged with the prologue applied.\n"
      << "  __shared__ float tile[" << colors << "][" << edge << "][" << edge << "];\n"
      << "  const int tile_x0 = (int)(blockIdx.x * " << tile_size << ") - " << radius << ";\n"
      << "  const int tile_y0 = (int)(blockIdx.y * " << tile_size << ") - " << radius << ";\n"
      << "  for (int k = (int)(threadIdx.y * " << tile_size << " + threadIdx.x); k < "
      << edge * edge << "; k += " << tile_size * tile_size << ") {\n"
      << "    const int ty = k / " << edge << ";\n"
      << "    const int tx = k - ty * " << edge << ";\n"
      << "    float t[" << colors << "];\n"
      << "    " << name << "_tap(src, tile_x0 + tx, tile_y0 + ty, width, height, t);\n";
  for (int c = 0; c < colors; ++c) {
    out << "    tile[" << c << "][ty][tx] = t[" << c << "];\n";
  }
  // The barrier comes BEFORE the bounds check, and that order is a correctness rule, not
  // a style: a thread whose output pixel is past the image edge still owns apron cells,
  // and a thread that returned early would leave them unloaded and never arrive here.
  out << "  }\n"
      << "  // Every thread arrives here, including those whose output pixel is past the\n"
      << "  // image edge — they still own apron cells. So the bounds check comes after.\n"
      << "  __syncthreads();\n";
}

// One neighbour's samples into `t`, at offset (dx, dy) from this thread's pixel. Naive:
// a clamped global load with the prologue recomputed, once per tap. Tiled: a read out
// of the staged tile. Everything around this fetch is the same text in both variants,
// which is what keeps tiling a memory optimization rather than a numerics change.
void emit_fetch(std::ostringstream& out, const std::string& name, int colors, bool tiled,
                const std::string& tap_args) {
  out << "      float t[" << colors << "];\n";
  if (!tiled) {
    out << "      " << name << "_tap(src, x + dx, y + dy, width, height, t" << tap_args
        << ");\n";
    return;
  }
  for (int c = 0; c < colors; ++c) {
    out << "      t[" << c << "] = tile[" << c << "][ly + dy][lx + dx];\n";
  }
}

// The stencil body. Accumulation order matches the oracle exactly: dy outer, dx inner,
// and the 2D weight formed as (wy * wx) before it multiplies the sample — the oracle's
// `sum += wy * wx * sample` associates the same way. Only FMA contraction is left to
// separate them, which is what the <=1 LSB tolerance is for.
//
// `runtime_radius` is a parameterized gaussian: its radius and weights are the kernel's
// `radius` and `taps` arguments rather than literals, so the tap loop's bounds are
// dynamic and NVRTC cannot unroll it — the cost the Phase 8 A/B measures. `tap_args`
// passes a parameterized prologue's values on to the tap helper.
void emit_stencil_body(std::ostringstream& out, const std::string& name, const Op& stencil,
                       int radius, int colors, bool tiled, bool runtime_radius,
                       const std::string& tap_args) {
  const std::string radius_text = runtime_radius ? "radius" : std::to_string(radius);
  const std::string weights = runtime_radius ? "taps.w" : name + "_w";

  if (tiled) {
    // No clamping from here on: the apron already holds every neighbour, edge-clamped
    // when it was loaded, so every (lx + dx, ly + dy) below is inside the tile.
    out << "  const int lx = (int)threadIdx.x + " << radius_text << ";  // this pixel, in tile\n"
        << "  const int ly = (int)threadIdx.y + " << radius_text << ";  // coordinates\n";
  }

  if (stencil.kind == OpKind::kGaussian) {
    out << "  float acc[" << colors << "] = " << zero_initializer(colors) << ";\n"
        << "  for (int dy = -" << radius_text << "; dy <= " << radius_text << "; ++dy) {\n"
        << "    for (int dx = -" << radius_text << "; dx <= " << radius_text << "; ++dx) {\n";
    emit_fetch(out, name, colors, tiled, tap_args);
    out << "      const float w = " << weights << "[dy + " << radius_text << "] * " << weights
        << "[dx + " << radius_text << "];\n";
    for (int c = 0; c < colors; ++c) {
      out << "      acc[" << c << "] += w * t[" << c << "];\n";
    }
    out << "    }\n  }\n";
    for (int c = 0; c < colors; ++c) {
      out << "  v[" << c << "] = imgjit_quantize(acc[" << c << "]);\n";
    }
    return;
  }

  out << "  float gx[" << colors << "] = " << zero_initializer(colors) << ";\n"
      << "  float gy[" << colors << "] = " << zero_initializer(colors) << ";\n"
      << "  for (int dy = -1; dy <= 1; ++dy) {\n"
      << "    for (int dx = -1; dx <= 1; ++dx) {\n";
  emit_fetch(out, name, colors, tiled, tap_args);
  out << "      const int k = (dy + 1) * 3 + (dx + 1);\n";
  for (int c = 0; c < colors; ++c) {
    out << "      gx[" << c << "] += " << name << "_gx[k] * t[" << c << "];\n"
        << "      gy[" << c << "] += " << name << "_gy[k] * t[" << c << "];\n";
  }
  out << "    }\n  }\n";
  for (int c = 0; c < colors; ++c) {
    out << "  v[" << c << "] = imgjit_quantize(sqrtf(gx[" << c << "] * gx[" << c << "] + gy[" << c
        << "] * gy[" << c << "]));\n";
  }
}

// `tile_size` is 0 for the naive variant.
GeneratedStage emit_stage(std::ostringstream& out, const std::string& name, const Stage& stage,
                          int channels, int colors, int tile_size, bool parameterized) {
  GeneratedStage generated;
  generated.kernel_name = name;
  generated.is_stencil = stage.has_stencil;

  // Of the two stencils only gaussian has a parameter. Sobel's 3x3 is the op's
  // definition, so it stays a literal in either mode.
  const bool runtime_radius =
      parameterized && stage.has_stencil && stage.stencil.op.kind == OpKind::kGaussian;
  const std::vector<std::size_t> prologue_params = param_indices(stage.prologue, parameterized);
  const std::vector<std::size_t> epilogue_params = param_indices(stage.epilogue, parameterized);

  int radius = 0;
  if (stage.has_stencil) {
    if (!runtime_radius) {
      radius = emit_stencil_constants(out, name, stage.stencil.op);
    }
    emit_tap_function(out, name, stage, channels, colors, parameterized,
                      param_list(prologue_params, ", float p"));
  }

  // Only a stencil has neighbours to share, so a pointwise-only stage is naive in
  // either variant.
  const bool tiled = stage.has_stencil && tile_size > 0;
  const int block_dim = tiled ? tile_size : kNaiveBlockDim;
  generated.block_dim = block_dim;

  // Every stage starts with the same four arguments, so the executor can ping-pong two
  // buffers through the whole chain without knowing what any stage does; a
  // parameterized stage appends its values after them, in param_ops order. Dimensions
  // are parameters here and nowhere else — that is invariant 4 made structural.
  out << "\nextern \"C\" __global__ void ";
  if (tiled) {
    // Caps registers so that tile_size^2 threads always fit on an SM. Without it a 32x32
    // tile (1024 threads) can compile to more registers than one block may have, and
    // the launch fails with out-of-resources — a CudaError, so a context recreation on
    // every frame. Naive stages need no cap: 256 threads fit at any register count.
    out << "__launch_bounds__(" << block_dim * block_dim << ") ";
  }
  out << name << "(\n"
      << "    const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,\n"
      << "    int width, int height";
  if (runtime_radius) {
    out << ", int radius, imgjit_gaussian_taps taps";
    generated.param_ops.push_back(stage.stencil.index);
  }
  out << param_list(prologue_params, ", float p") << param_list(epilogue_params, ", float p")
      << ") {\n";
  generated.param_ops.insert(generated.param_ops.end(), prologue_params.begin(),
                             prologue_params.end());
  generated.param_ops.insert(generated.param_ops.end(), epilogue_params.begin(),
                             epilogue_params.end());
  if (tiled) {
    emit_tile_load(out, name, radius, tile_size, colors);
  }
  out << "  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
      << "  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);\n"
      << "  if (x >= width || y >= height) { return; }\n"
      << "  const int i = (y * width + x) * " << channels << ";\n"
      << "  float v[" << colors << "];\n";

  if (!stage.has_stencil) {
    for (int c = 0; c < colors; ++c) {
      out << "  v[" << c << "] = imgjit_load_sample(src[i + " << c << "]);\n";
    }
    emit_pointwise(out, stage.prologue, colors, "v", "  ", parameterized);
  } else {
    emit_stencil_body(out, name, stage.stencil.op, radius, colors, tiled, runtime_radius,
                      param_list(prologue_params, ", p"));
  }

  emit_pointwise(out, stage.epilogue, colors, "v", "  ", parameterized);

  for (int c = 0; c < colors; ++c) {
    out << "  dst[i + " << c << "] = imgjit_store_sample(v[" << c << "]);\n";
  }
  if (channels == 4) {
    // Read from the centre pixel of `src`, not from the stencil, so opacity is never
    // blurred or inverted (imgjit/core/op_chain.h).
    out << "  dst[i + 3] = src[i + 3];  // alpha passes through every op untouched\n";
  }
  out << "}\n";
  return generated;
}

}  // namespace

bool is_supported_tile(const TileVariant tile, const int tile_size) {
  if (tile == TileVariant::kNaive) {
    return tile_size == 0;
  }
  return tile == TileVariant::kTiled && tile_size >= 1 && tile_size <= kMaxTileSize;
}

bool is_supported_constants(const ConstantsMode constants, const TileVariant tile) {
  if (constants == ConstantsMode::kBaked) {
    return true;
  }
  return constants == ConstantsMode::kParameterized && tile == TileVariant::kNaive;
}

GeneratedProgram emit_cuda_source(const KernelKey& key) {
  if (!is_supported_channel_count(key.channels)) {
    throw std::invalid_argument("emit_cuda_source: unsupported channel count " +
                                std::to_string(key.channels));
  }
  if (!is_supported_tile(key.tile, key.tile_size)) {
    throw std::invalid_argument("emit_cuda_source: unsupported tile size " +
                                std::to_string(key.tile_size) +
                                " (naive takes 0, tiled takes 1-" + std::to_string(kMaxTileSize) +
                                ")");
  }
  if (!is_supported_constants(key.constants, key.tile)) {
    throw std::invalid_argument(
        "emit_cuda_source: parameterized constants need naive stencils — a tiled stage "
        "sizes its __shared__ array from the stencil radius, which is only known at launch");
  }

  const int colors = cpu::color_channels(key.channels);
  const std::vector<Stage> stages = plan_stages(key.chain);
  const int tile_size = key.tile == TileVariant::kTiled ? key.tile_size : 0;
  const bool parameterized = key.constants == ConstantsMode::kParameterized;

  std::ostringstream out;
  out << "// Generated by imgjit codegen (src/backend/cuda/codegen.cpp) — do not edit.\n"
      << "// chain:    \"" << chain_comment(key.chain, parameterized) << "\"\n"
      << "// channels: " << key.channels << " (" << colors << " colour + "
      << (key.channels == 4 ? 1 : 0) << " alpha)\n"
      << "// stages:   " << stages.size() << "\n";
  if (tile_size == 0) {
    out << "// tile:     naive (every tap is a global load)\n";
  } else {
    out << "// tile:     " << tile_size << "x" << tile_size
        << " shared-memory tile on stencil stages\n";
  }
  if (parameterized) {
    out << "// constants: parameterized — every op parameter is a kernel argument, so this\n"
        << "// source serves every value of them (imgjit/core/kernel_key.h).\n";
  }
  out << "//\n"
      << "// Width and height are absent from this source by construction: they are\n"
      << "// launch arguments, so one compile serves every resolution (invariant 4).\n"
      << kKernelPrelude;
  if (parameterized) {
    // Passed by value, so it lands in the kernel's parameter space like any scalar
    // argument. Sized for the widest gaussian; a narrower one leaves the tail unread.
    out << "\nstruct imgjit_gaussian_taps { float w[" << kMaxGaussianTaps << "]; };\n";
  }

  GeneratedProgram program;
  program.stages.reserve(stages.size());
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const std::string name = kStagePrefix + std::to_string(i);
    program.stages.push_back(
        emit_stage(out, name, stages[i], key.channels, colors, tile_size, parameterized));
  }

  program.source = out.str();
  return program;
}

}  // namespace imgjit::cuda
