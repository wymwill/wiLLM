// Golden tests: run the C++ forward pass and compare every recorded
// activation against the numpy reference in tools/llama2c.py.
//
// The tiny model (tests/data) always runs. stories15M runs when
// models/stories15M.bin and models/stories15M.golden exist, see tools/README.md.

#include "willm/model/transformer.h"

#include <doctest/doctest.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "golden.h"

namespace fs = std::filesystem;
using willm::core::ConstTensorView;

namespace {

// Absolute plus relative tolerance: |actual - expected| <= kTol * (1 + |expected|).
constexpr double kTol = 1e-4;

// Copies every activation the forward pass records.
class RecordingSink : public willm::model::ActivationSink {
public:
    void record(std::string_view name, ConstTensorView value) override {
        const float* p = value.data<float>();
        values_[std::string(name)] = std::vector<float>(p, p + value.numel());
    }
    const std::vector<float>* find(const std::string& name) const {
        const auto it = values_.find(name);
        return it == values_.end() ? nullptr : &it->second;
    }
    void clear() { values_.clear(); }

private:
    std::map<std::string, std::vector<float>> values_;
};

// Compares one position. Reports only the first activation that differs,
// since everything after it will differ too.
void compare_position(const willm::testing::GoldenFile& golden, int pos,
                      const RecordingSink& sink) {
    const std::string prefix = "pos" + std::to_string(pos) + ".";
    for (const auto& full_name : golden.names()) {
        if (!full_name.starts_with(prefix)) {
            continue;
        }
        const std::string name = full_name.substr(prefix.size());
        const ConstTensorView expected = golden.get(full_name);
        const std::vector<float>* actual = sink.find(name);
        INFO("position " << pos << ", activation \"" << name << "\"");
        REQUIRE_MESSAGE(actual != nullptr, "forward() did not record this activation");
        REQUIRE(static_cast<std::int64_t>(actual->size()) == expected.numel());

        const float* e = expected.data<float>();
        for (std::size_t i = 0; i < actual->size(); ++i) {
            const double diff = std::abs(static_cast<double>((*actual)[i]) - e[i]);
            if (!(diff <= kTol * (1.0 + std::abs(static_cast<double>(e[i]))))) {
                FAIL("element " << i << ": got " << (*actual)[i] << ", expected " << e[i]);
            }
        }
    }
}

void run_golden(const fs::path& model_path, const fs::path& golden_path) {
    const auto model = willm::model::load_llama2c(model_path);
    const auto golden = willm::testing::GoldenFile::open(golden_path);
    willm::model::Transformer transformer(model);
    auto cache = transformer.make_cache();

    const ConstTensorView tokens = golden.get("tokens");
    RecordingSink sink;
    for (int pos = 0; pos < tokens.numel(); ++pos) {
        const auto token = static_cast<std::int32_t>(tokens.data<float>()[pos]);
        sink.clear();
        const ConstTensorView logits = transformer.forward(token, pos, cache, &sink);
        CHECK(logits.numel() == model.config.vocab_size);
        compare_position(golden, pos, sink);
    }
}

const fs::path kData = WILLM_TEST_DATA_DIR;
const fs::path kModels = WILLM_MODELS_DIR;
const bool kHaveStories =
    fs::exists(kModels / "stories15M.bin") && fs::exists(kModels / "stories15M.golden");

} // namespace

TEST_CASE("golden: tiny model, grouped query attention, separate classifier") {
    run_golden(kData / "tiny.bin", kData / "tiny.golden");
}

TEST_CASE("golden: stories15M" * doctest::skip(!kHaveStories)) {
    run_golden(kModels / "stories15M.bin", kModels / "stories15M.golden");
}
