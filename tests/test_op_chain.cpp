// Parser and canonicalizer (docs/PLAN.md Phase 2).

#include "catch_amalgamated.hpp"
#include "imgjit/core/op_chain.h"

using imgjit::canonical_string;
using imgjit::Op;
using imgjit::OpKind;
using imgjit::parse_op_chain;

namespace {

// Reparsing a canonical string must produce the same chain — the property the kernel
// cache leans on when it is keyed by canonical form.
std::string canonicalize(std::string_view text) {
  const auto chain = parse_op_chain(text);
  REQUIRE(chain.has_value());
  const std::string canonical = canonical_string(*chain);
  const auto reparsed = parse_op_chain(canonical);
  REQUIRE(reparsed.has_value());
  REQUIRE(canonical_string(*reparsed) == canonical);
  return canonical;
}

}  // namespace

TEST_CASE("parser accepts the closed op set", "[core]") {
  const auto chain = parse_op_chain("grayscale,gaussian:1.4,sobel,threshold:0.3");
  REQUIRE(chain.has_value());
  REQUIRE(chain->ops.size() == 4);
  CHECK(chain->ops[0] == Op{OpKind::kGrayscale, 0.0F});
  CHECK(chain->ops[1].kind == OpKind::kGaussian);
  CHECK(chain->ops[1].param == Catch::Approx(1.4F));
  CHECK(chain->ops[2].kind == OpKind::kSobel);
  CHECK(chain->ops[3].param == Catch::Approx(0.3F));
}

TEST_CASE("an empty chain is a valid identity pass", "[core]") {
  const auto chain = parse_op_chain("");
  REQUIRE(chain.has_value());
  CHECK(chain->ops.empty());
  CHECK(canonical_string(*chain).empty());
  CHECK(parse_op_chain("   ").has_value());
}

TEST_CASE("whitespace around names, colons and parameters is ignored", "[core]") {
  CHECK(canonicalize("  grayscale , gaussian : 1.4 ,sobel  ") == "grayscale,gaussian:1.4,sobel");
}

TEST_CASE("omitted parameters take the op's documented default", "[core]") {
  CHECK(canonicalize("gaussian") == "gaussian:1.0");
  CHECK(canonicalize("threshold") == "threshold:0.5");
  CHECK(canonicalize("brightness") == "brightness:0.0");
}

TEST_CASE("float formatting is normalized and parameters quantized to 2 decimals",
          "[core]") {
  CHECK(canonicalize("gaussian:1.40") == "gaussian:1.4");
  CHECK(canonicalize("gaussian:1.4000001") == "gaussian:1.4");
  CHECK(canonicalize("gaussian:1.400") == canonicalize("gaussian:1.4"));
  CHECK(canonicalize("gaussian:1.45") == "gaussian:1.45");
  // Quantization is lossy by design: 1.454 and 1.446 are one kernel, not three.
  CHECK(canonicalize("gaussian:1.454") == "gaussian:1.45");
  CHECK(canonicalize("gaussian:1.446") == "gaussian:1.45");
  CHECK(canonicalize("brightness:-0.20") == "brightness:-0.2");
  CHECK(canonicalize("brightness:0") == "brightness:0.0");
  CHECK(canonicalize("threshold:1") == "threshold:1.0");
}

TEST_CASE("canonicalization never reorders ops", "[core]") {
  // gaussian->sobel and sobel->gaussian are different images. If the canonicalizer
  // ever sorts for a "nicer" key, this is the test that catches it.
  CHECK(canonicalize("gaussian:1.4,sobel") == "gaussian:1.4,sobel");
  CHECK(canonicalize("sobel,gaussian:1.4") == "sobel,gaussian:1.4");
  CHECK(canonicalize("invert,grayscale") == "invert,grayscale");
  CHECK(canonicalize("grayscale,invert") == "grayscale,invert");
}

TEST_CASE("a repeated op is kept, not deduplicated", "[core]") {
  CHECK(canonicalize("invert,invert") == "invert,invert");
}

TEST_CASE("malformed chains are rejected with a reason", "[core]") {
  const auto rejected = [](std::string_view text) {
    std::string error;
    const auto chain = parse_op_chain(text, &error);
    CHECK_FALSE(chain.has_value());
    CHECK_FALSE(error.empty());
  };

  rejected("sharpen");             // not in the closed set
  rejected("grayscale,");          // trailing comma
  rejected(",grayscale");          // leading comma
  rejected("grayscale,,sobel");    // empty op
  rejected("sobel:2");             // parameter on a paramless op
  rejected("grayscale:1");         // ditto
  rejected("gaussian:");           // colon with no value
  rejected("gaussian:abc");        // not a number
  rejected("gaussian:1.4x");       // trailing junk after a valid prefix
  rejected("gaussian:nan");        // non-finite
  rejected("gaussian:inf");
  rejected("GRAYSCALE");           // op names are case-sensitive
}

TEST_CASE("out-of-range parameters are rejected", "[core]") {
  const auto rejected = [](std::string_view text) {
    CHECK_FALSE(parse_op_chain(text).has_value());
  };

  // The gaussian bound is what keeps the baked stencil radius finite (radius =
  // ceil(3*sigma) <= 12); the others keep normalized math in range.
  rejected("gaussian:0");
  rejected("gaussian:0.05");
  rejected("gaussian:4.01");
  rejected("gaussian:-1");
  rejected("threshold:-0.01");
  rejected("threshold:1.01");
  rejected("brightness:-1.01");
  rejected("brightness:1.01");

  CHECK(parse_op_chain("gaussian:0.1").has_value());
  CHECK(parse_op_chain("gaussian:4").has_value());
  CHECK(parse_op_chain("threshold:0").has_value());
  CHECK(parse_op_chain("brightness:-1").has_value());
}

TEST_CASE("the op spec table is consistent with the OpKind enum", "[core]") {
  // op_spec() indexes the table by enum value, so a reordered table would silently
  // hand out the wrong name, bounds and default.
  for (const imgjit::OpSpec& spec : imgjit::kOpSpecs) {
    CHECK(imgjit::op_spec(spec.kind).name == spec.name);
    if (spec.takes_param) {
      CHECK(spec.default_param >= spec.min_param);
      CHECK(spec.default_param <= spec.max_param);
    }
  }
}
