#include "willm/core/check.h"
#include "willm/model/weights.h"

#include <doctest/doctest.h>

#include <unistd.h>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using willm::model::load_llama2c;

namespace {
const fs::path kTiny = fs::path(WILLM_TEST_DATA_DIR) / "tiny.bin";
const fs::path kStories = fs::path(WILLM_MODELS_DIR) / "stories15M.bin";
} // namespace

TEST_CASE("load_llama2c reads the tiny test model") {
    const auto m = load_llama2c(kTiny);
    const auto& c = m.config;
    CHECK(c.dim == 32);
    CHECK(c.hidden_dim == 64);
    CHECK(c.n_layers == 2);
    CHECK(c.n_heads == 4);
    CHECK(c.n_kv_heads == 2);
    CHECK(c.vocab_size == 32);
    CHECK(c.seq_len == 8);
    CHECK_FALSE(m.shared_classifier); // tiny.bin stores wcls separately

    const auto& w = m.weights;
    CHECK(w.wq.dim(0) == 2);
    CHECK(w.wk.dim(1) == 16); // kv_dim = 2 kv heads * head_dim 8
    CHECK(w.w2.dim(2) == 64); // w2 is [L, dim, hidden]
    CHECK(w.wcls.raw_data() != w.token_embedding.raw_data());
    // The last float of wcls is the last float of the file.
    CHECK(w.wcls.raw_data() + w.wcls.nbytes() == m.file.data() + m.file.size());
}

TEST_CASE("load_llama2c rejects a truncated file") {
    const fs::path cut =
        fs::temp_directory_path() / ("willm_" + std::to_string(::getpid()) + "_truncated.bin");
    fs::copy_file(kTiny, cut, fs::copy_options::overwrite_existing);
    fs::resize_file(cut, fs::file_size(kTiny) - 4);
    CHECK_THROWS_AS(load_llama2c(cut), willm::core::CheckError);
    fs::remove(cut);
}

TEST_CASE("load_llama2c reads stories15M" * doctest::skip(!fs::exists(kStories))) {
    const auto m = load_llama2c(kStories);
    CHECK(m.config.dim == 288);
    CHECK(m.config.n_layers == 6);
    CHECK(m.config.vocab_size == 32000);
    CHECK(m.shared_classifier);
}
